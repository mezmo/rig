use futures::{FutureExt, StreamExt, TryStreamExt, channel::oneshot::Canceled, stream};
use tokio::sync::mpsc::{Sender, error::SendError};

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{Tool, ToolDyn, ToolError, ToolSet, ToolSetError, ToolType, call_tool_named},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter},
};

pub struct ToolServer {
    /// A list of static tool names.
    /// These tools will always exist on the tool server for as long as they are not deleted.
    static_tool_names: Vec<String>,
    /// Dynamic tools. These tools will be dynamically fetched from a given vector store.
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
    /// The toolset where tools are called (to be executed).
    toolset: ToolSet,
}

impl Default for ToolServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolServer {
    pub fn new() -> Self {
        Self {
            static_tool_names: Vec::new(),
            dynamic_tools: Vec::new(),
            toolset: ToolSet::default(),
        }
    }

    pub(crate) fn static_tool_names(mut self, names: Vec<String>) -> Self {
        self.static_tool_names = names;
        self
    }

    pub(crate) fn add_tools(mut self, tools: ToolSet) -> Self {
        self.toolset = tools;
        self
    }

    pub(crate) fn add_dynamic_tools(
        mut self,
        dyn_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
    ) -> Self {
        self.dynamic_tools = dyn_tools;
        self
    }

    /// Add a static tool to the agent
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.toolset.add_tool(tool);
        self.static_tool_names.push(toolname);
        self
    }

    // Add an MCP tool (from `rmcp`) to the agent
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(feature = "rmcp")]
    pub fn rmcp_tool(mut self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> Self {
        use crate::tool::rmcp::McpTool;
        let toolname = tool.name.clone();
        self.toolset
            .add_tool(McpTool::from_mcp_server(tool, client));
        self.static_tool_names.push(toolname.to_string());
        self
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.dynamic_tools.push((sample, Box::new(dynamic_tools)));
        self.toolset.add_tools(toolset);
        self
    }

    pub fn run(mut self) -> ToolServerHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1000);

        #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
        tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                self.handle_message(message).await;
            }
        });

        // SAFETY: `rig` currently doesn't compile to WASM without the `worker` feature.
        // Therefore, we can safely assume that the user won't try to compile to wasm without the worker feature.
        #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
        wasm_bindgen_futures::spawn_local(async move {
            while let Some(message) = rx.recv().await {
                self.handle_message(message).await;
            }
        });

        ToolServerHandle(tx)
    }

    pub async fn handle_message(&mut self, message: ToolServerRequest) {
        let ToolServerRequest {
            callback_channel,
            data,
        } = message;

        match data {
            ToolServerRequestMessageKind::AddTool(tool) => {
                self.static_tool_names.push(tool.name());
                self.toolset.add_tool_boxed(tool);
                callback_channel
                    .send(ToolServerResponse::ToolAdded)
                    .unwrap();
            }
            ToolServerRequestMessageKind::AppendToolset(tools) => {
                self.toolset.add_tools(tools);
                callback_channel
                    .send(ToolServerResponse::ToolAdded)
                    .unwrap();
            }
            ToolServerRequestMessageKind::RemoveTool { tool_name } => {
                self.static_tool_names.retain(|x| *x != tool_name);
                self.toolset.delete_tool(&tool_name);
                callback_channel
                    .send(ToolServerResponse::ToolDeleted)
                    .unwrap();
            }
            ToolServerRequestMessageKind::GetTool { name } => {
                let tool = self.toolset.get(&name).cloned().map(ResolvedTool);
                let _ = callback_channel.send(ToolServerResponse::Tool(tool));
            }
            ToolServerRequestMessageKind::GetToolDefs { prompt } => {
                let res = self.get_tool_definitions(prompt).await.unwrap();
                callback_channel
                    .send(ToolServerResponse::ToolDefinitions(res))
                    .unwrap();
            }
        }
    }

    pub async fn get_tool_definitions(
        &mut self,
        text: Option<String>,
    ) -> Result<Vec<ToolDefinition>, CompletionError> {
        let static_tool_names = self.static_tool_names.clone();
        let mut tools = if let Some(text) = text {
            stream::iter(self.dynamic_tools.iter())
                        .then(|(num_sample, index)| async {
                            let req =
                                VectorSearchRequest::builder()
                                    .query(text.clone())
                                    .samples(*num_sample as u64)
                                    .build()
                                    .expect("Creating VectorSearchRequest here shouldn't fail since the query and samples to return are always present");
                            Ok::<_, VectorStoreError>(
                        index.as_ref()
                                    .top_n_ids(req.map_filter(Filter::interpret))
                                    .await?
                                    .into_iter()
                                    .map(|(_, id)| id)
                                    .collect::<Vec<String>>(),
                            )
                        })
                        .try_fold(vec![], |mut acc, docs| async {
                            for doc in docs {
                                if let Some(tool) = self.toolset.get(&doc) {
                                    acc.push(tool.definition(text.clone()).await)
                                } else {
                                    tracing::warn!("Tool implementation not found in toolset: {}", doc);
                                }
                            }
                            Ok(acc)
                        })
                        .await
                        .map_err(|e| CompletionError::RequestError(Box::new(e)))?
        } else {
            Vec::new()
        };

        for toolname in static_tool_names {
            if let Some(tool) = self.toolset.get(&toolname) {
                tools.push(tool.definition(String::new()).await)
            } else {
                tracing::warn!("Tool implementation not found in toolset: {}", toolname);
            }
        }

        Ok(tools)
    }
}

#[derive(Clone)]
pub struct ToolServerHandle(Sender<ToolServerRequest>);

impl ToolServerHandle {
    pub async fn add_tool(&self, tool: impl ToolDyn + 'static) -> Result<(), ToolServerError> {
        let tool = Box::new(tool);

        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::AddTool(tool),
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolAdded = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::AppendToolset(toolset),
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolAdded = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    /// Removes `tool_name` for new calls. A call already running keeps the
    /// tool until it finishes, and this does not wait for it.
    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::RemoveTool {
                    tool_name: tool_name.to_string(),
                },
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolDeleted = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    /// Runs `tool_name` in the caller's task, so the call sees the caller's
    /// task-locals and span, and calls from different callers run at the same
    /// time. The server only hands over the tool, and it handles one request at
    /// a time, so the lookup can wait behind a tool-definitions fetch.
    pub async fn call_tool(&self, tool_name: &str, args: &str) -> Result<String, ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::GetTool {
                    name: tool_name.to_string(),
                },
            })
            .await?;

        let tool = match rx.await? {
            ToolServerResponse::Tool(tool) => tool,
            invalid => return Err(ToolServerError::InvalidMessage(invalid)),
        };

        // A panicking tool fails its own call. The caller's run and the
        // server carry on.
        let call = call_tool_named(tool_name, tool.as_ref().map(|t| &t.0), args.to_string());
        std::panic::AssertUnwindSafe(call)
            .catch_unwind()
            .await
            .unwrap_or_else(|panic| {
                Err(ToolSetError::ToolCallError(ToolError::ToolCallError(
                    format!("tool `{tool_name}` panicked: {}", panic_message(&panic)).into(),
                )))
            })
            .map_err(|err| {
                // Callers match on this text. The tool set's error is
                // stringified inside a tool call error.
                ToolServerError::ToolsetError(ToolSetError::ToolCallError(
                    ToolError::ToolCallError(err.to_string().into()),
                ))
            })
    }

    pub async fn get_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::GetToolDefs { prompt },
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolDefinitions(tooldefs) = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(tooldefs)
    }
}

/// Returns the text a panic carried, or "no message" when it carried none.
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
}

pub struct ToolServerRequest {
    callback_channel: futures::channel::oneshot::Sender<ToolServerResponse>,
    data: ToolServerRequestMessageKind,
}

pub enum ToolServerRequestMessageKind {
    AddTool(Box<dyn ToolDyn>),
    AppendToolset(ToolSet),
    RemoveTool { tool_name: String },
    GetTool { name: String },
    GetToolDefs { prompt: Option<String> },
}

#[derive(Debug)]
pub enum ToolServerResponse {
    ToolAdded,
    ToolDeleted,
    Tool(Option<ResolvedTool>),
    ToolDefinitions(Vec<ToolDefinition>),
}

/// A tool the server handed to a caller to run.
pub struct ResolvedTool(ToolType);

impl std::fmt::Debug for ResolvedTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ResolvedTool").field(&self.0.name()).finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    #[error("Sending message was cancelled")]
    Canceled(#[from] Canceled),
    #[error("Toolset error: {0}")]
    ToolsetError(#[from] ToolSetError),
    #[error("Error while sending message: {0}")]
    SendError(#[from] SendError<ToolServerRequest>),
    #[error("An invalid message type was returned")]
    InvalidMessage(ToolServerResponse),
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use crate::{
        completion::ToolDefinition,
        tool::{Tool, server::ToolServer},
    };

    #[derive(Deserialize)]
    struct OperationArgs {
        x: i32,
        y: i32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Math error")]
    struct MathError;

    #[derive(Deserialize, Serialize)]
    struct Adder;
    impl Tool for Adder {
        const NAME: &'static str = "add";
        type Error = MathError;
        type Args = OperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "add".to_string(),
                description: "Add x and y together".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The first number to add"
                        },
                        "y": {
                            "type": "number",
                            "description": "The second number to add"
                        }
                    },
                    "required": ["x", "y"],
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            println!("[tool-call] Adding {} and {}", args.x, args.y);
            let result = args.x + args.y;
            Ok(result)
        }
    }

    struct ReadsCaller;

    tokio::task_local! {
        static CALLER: &'static str;
    }

    impl Tool for ReadsCaller {
        const NAME: &'static str = "reads_caller";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: String::new(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(CALLER
                .try_with(|caller| caller.to_string())
                .unwrap_or("none".into()))
        }
    }

    /// Returns once two calls are inside it at the same time.
    struct Meets(std::sync::Arc<tokio::sync::Barrier>);

    impl Tool for Meets {
        const NAME: &'static str = "meets";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: String::new(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.0.wait().await;
            Ok("met".into())
        }
    }

    struct Fails;

    #[derive(Debug, thiserror::Error)]
    #[error("Connection refused")]
    struct Refused;

    impl Tool for Fails {
        const NAME: &'static str = "fails";
        type Error = Refused;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: String::new(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Err(Refused)
        }
    }

    struct Panics;

    impl Tool for Panics {
        const NAME: &'static str = "panics";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: String::new(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            panic!("tool blew up")
        }
    }

    #[tokio::test]
    async fn a_panicking_tool_fails_only_its_call() {
        let handle = ToolServer::new().tool(Panics).tool(Adder).run();

        let failed = handle.call_tool("panics", "{}").await.unwrap_err();
        assert_eq!(
            failed.to_string(),
            "Toolset error: ToolCallError: ToolCallError: ToolCallError: \
             tool `panics` panicked: tool blew up"
        );

        let next = handle.call_tool("add", r#"{"x": 1, "y": 2}"#).await;
        assert_eq!(next.unwrap(), "3", "the server keeps serving calls");
    }

    struct Slow(std::sync::Arc<std::sync::atomic::AtomicBool>);

    impl Tool for Slow {
        const NAME: &'static str = "slow";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: String::new(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok("done".into())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_caller_stops_the_tool() {
        let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let handle = ToolServer::new().tool(Slow(finished.clone())).run();

        let gave_up = tokio::time::timeout(
            std::time::Duration::from_millis(10),
            handle.call_tool("slow", "{}"),
        )
        .await;
        assert!(gave_up.is_err(), "the caller stops waiting first");

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(
            !finished.load(std::sync::atomic::Ordering::SeqCst),
            "a dropped call does not run to its end"
        );
    }

    #[tokio::test]
    async fn a_call_sees_the_callers_task_locals() {
        let handle = ToolServer::new().tool(ReadsCaller).run();

        let seen = CALLER
            .scope("run-1", handle.call_tool("reads_caller", "{}"))
            .await
            .unwrap();

        assert_eq!(seen, "\"run-1\"");
    }

    #[tokio::test]
    async fn calls_from_different_tasks_run_at_the_same_time() {
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));
        let handle = ToolServer::new().tool(Meets(barrier)).run();
        let call_from_a_task = || {
            let handle = handle.clone();
            tokio::spawn(async move { handle.call_tool("meets", "{}").await })
        };
        let (first, second) = (call_from_a_task(), call_from_a_task());

        let both = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            (first.await.unwrap(), second.await.unwrap())
        })
        .await
        .expect("the second call must not wait for the first to finish");

        assert_eq!(both.0.unwrap(), "\"met\"");
        assert_eq!(both.1.unwrap(), "\"met\"");
    }

    #[tokio::test]
    async fn a_failed_call_keeps_its_error_text() {
        let handle = ToolServer::new().tool(Fails).run();

        let failed = handle.call_tool("fails", "{}").await.unwrap_err();
        assert_eq!(
            failed.to_string(),
            "Toolset error: ToolCallError: ToolCallError: ToolCallError: Connection refused"
        );

        let missing = handle.call_tool("missing", "{}").await.unwrap_err();
        assert_eq!(
            missing.to_string(),
            "Toolset error: ToolCallError: ToolCallError: ToolNotFoundError: missing"
        );
    }

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(Adder).await.unwrap();
        let res = handle.get_tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 1);

        let json_args_as_string =
            serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let res = handle.call_tool("add", &json_args_as_string).await.unwrap();
        assert_eq!(res, "7");

        handle.remove_tool("add").await.unwrap();
        let res = handle.get_tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 0);
    }
}
