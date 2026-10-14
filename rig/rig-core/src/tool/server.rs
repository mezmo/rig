use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use futures::{FutureExt, StreamExt, TryStreamExt, stream};

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{Tool, ToolDyn, ToolSet, ToolSetError, ToolType, call_tool_named, panic_message},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter},
};

#[derive(Clone)]
pub struct ToolServer {
    /// A list of static tool names.
    /// These tools will always exist on the tool server for as long as they are not deleted.
    static_tool_names: Vec<String>,
    /// Dynamic tools. These tools will be dynamically fetched from a given vector store.
    dynamic_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
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
        self.dynamic_tools = dyn_tools
            .into_iter()
            .map(|(sample, index)| (sample, Arc::from(index)))
            .collect();
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
        self.dynamic_tools.push((sample, Arc::new(dynamic_tools)));
        self.toolset.add_tools(toolset);
        self
    }

    pub fn run(self) -> ToolServerHandle {
        ToolServerHandle(Arc::new(RwLock::new(self)))
    }

    pub async fn get_tool_definitions(
        &self,
        text: Option<String>,
    ) -> Result<Vec<ToolDefinition>, CompletionError> {
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

        for toolname in &self.static_tool_names {
            if let Some(tool) = self.toolset.get(toolname) {
                tools.push(tool.definition(String::new()).await)
            } else {
                tracing::warn!("Tool implementation not found in toolset: {}", toolname);
            }
        }

        Ok(tools)
    }
}

#[derive(Clone)]
pub struct ToolServerHandle(Arc<RwLock<ToolServer>>);

impl ToolServerHandle {
    fn read(&self) -> RwLockReadGuard<'_, ToolServer> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Held only to move tools in and out of the map. A tool's own code, its
    /// name and its drop included, runs after the lock is released.
    fn write(&self) -> RwLockWriteGuard<'_, ToolServer> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }

    pub async fn add_tool(&self, tool: impl ToolDyn + 'static) -> Result<(), ToolServerError> {
        let name = tool.name();
        let tool = ToolType::Simple(Arc::new(tool));
        let replaced = {
            let mut server = self.write();
            server.static_tool_names.push(name.clone());
            server.toolset.tools.insert(name, tool)
        };
        drop(replaced);
        Ok(())
    }

    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let replaced: Vec<ToolType> = {
            let mut server = self.write();
            toolset
                .tools
                .into_iter()
                .filter_map(|(name, tool)| server.toolset.tools.insert(name, tool))
                .collect()
        };
        drop(replaced);
        Ok(())
    }

    /// Removes `tool_name` for new calls. A call already running keeps the
    /// tool until it finishes, and this does not wait for it.
    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let removed = {
            let mut server = self.write();
            server.static_tool_names.retain(|name| name != tool_name);
            server.toolset.tools.remove(tool_name)
        };
        drop(removed);
        Ok(())
    }

    /// Runs `tool_name` in the caller's task, so the call sees the caller's
    /// task-locals and span, and calls from different callers run at the same
    /// time.
    ///
    /// A panicking tool returns an `Err` with the panic's message. On wasm a
    /// panic aborts instead.
    pub async fn call_tool(&self, tool_name: &str, args: &str) -> Result<String, ToolServerError> {
        let tool = self.read().toolset.get(tool_name).cloned();
        Ok(call_tool_named(tool_name, tool.as_ref(), args.to_string()).await?)
    }

    /// Fetches tool definitions from a copy of the server taken under the
    /// lock, so the fetch never holds the lock while it awaits.
    pub async fn get_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        let server = self.read().clone();

        // A panicking definition or store fails this fetch, not the caller.
        std::panic::AssertUnwindSafe(server.get_tool_definitions(prompt))
            .catch_unwind()
            .await
            .unwrap_or_else(|panic| {
                Err(CompletionError::RequestError(
                    format!("tool definitions panicked: {}", panic_message(&panic)).into(),
                ))
            })
            .map_err(ToolServerError::DefinitionsFailed)
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolServerError {
    #[error("Toolset error: {0}")]
    ToolsetError(#[from] ToolSetError),
    #[error("Tool definitions failed: {0}")]
    DefinitionsFailed(#[source] CompletionError),
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::Duration;

    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use tokio::sync::Barrier;

    use crate::{
        completion::ToolDefinition,
        tool::{
            Tool, ToolError, ToolSet, ToolSetError,
            server::{ToolServer, ToolServerError},
        },
        vector_store::{
            TopNResults, VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn,
            request::Filter,
        },
        wasm_compat::WasmBoxedFuture,
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

    fn bare_definition(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: String::new(),
            parameters: json!({"type": "object"}),
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
            bare_definition(Self::NAME)
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(CALLER
                .try_with(|caller| caller.to_string())
                .unwrap_or("none".into()))
        }
    }

    /// Returns once two calls are inside it at the same time.
    struct Meets(Arc<Barrier>);

    impl Tool for Meets {
        const NAME: &'static str = "meets";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            bare_definition(Self::NAME)
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
            bare_definition(Self::NAME)
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
            bare_definition(Self::NAME)
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
            "Toolset error: ToolCallError: ToolCallError: \
             tool `panics` panicked: tool blew up"
        );

        let next = handle.call_tool("add", r#"{"x": 1, "y": 2}"#).await;
        assert_eq!(next.unwrap(), "3", "the handle keeps serving calls");
    }

    struct Slow(Arc<AtomicBool>);

    impl Tool for Slow {
        const NAME: &'static str = "slow";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            bare_definition(Self::NAME)
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            tokio::time::sleep(Duration::from_millis(100)).await;
            self.0.store(true, Ordering::SeqCst);
            Ok("done".into())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_caller_stops_the_tool() {
        let finished = Arc::new(AtomicBool::new(false));
        let handle = ToolServer::new().tool(Slow(finished.clone())).run();

        let gave_up =
            tokio::time::timeout(Duration::from_millis(10), handle.call_tool("slow", "{}")).await;
        assert!(gave_up.is_err(), "the caller stops waiting first");

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            !finished.load(Ordering::SeqCst),
            "a dropped call does not run to its end"
        );
    }

    struct SlowDefinition;

    impl Tool for SlowDefinition {
        const NAME: &'static str = "slow_definition";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            tokio::time::sleep(Duration::from_millis(100)).await;
            bare_definition(Self::NAME)
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("ok".into())
        }
    }

    struct PanickyDefinition;

    impl Tool for PanickyDefinition {
        const NAME: &'static str = "panicky_definition";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            panic!("definition blew up")
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("ok".into())
        }
    }

    struct FailingIndex;

    impl VectorStoreIndexDyn for FailingIndex {
        fn top_n<'a>(
            &'a self,
            _req: VectorSearchRequest<Filter<serde_json::Value>>,
        ) -> WasmBoxedFuture<'a, TopNResults> {
            Box::pin(async { Err(store_down()) })
        }

        fn top_n_ids<'a>(
            &'a self,
            _req: VectorSearchRequest<Filter<serde_json::Value>>,
        ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
            Box::pin(async { Err(store_down()) })
        }
    }

    fn store_down() -> VectorStoreError {
        VectorStoreError::DatastoreError("store is down".into())
    }

    #[tokio::test(start_paused = true)]
    async fn a_caller_that_stops_waiting_leaves_the_handle_working() {
        let handle = ToolServer::new().tool(SlowDefinition).tool(Adder).run();

        let gave_up =
            tokio::time::timeout(Duration::from_millis(10), handle.get_tool_defs(None)).await;
        assert!(gave_up.is_err(), "the caller stops waiting first");

        let next = handle.call_tool("add", r#"{"x": 1, "y": 2}"#).await;
        assert_eq!(next.unwrap(), "3", "the handle keeps serving calls");
    }

    #[tokio::test(start_paused = true)]
    async fn a_call_does_not_wait_behind_a_definitions_fetch() {
        let handle = ToolServer::new().tool(SlowDefinition).tool(Adder).run();
        let fetch = tokio::spawn({
            let handle = handle.clone();
            async move { handle.get_tool_defs(None).await }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;

        let call = tokio::time::timeout(
            Duration::from_millis(50),
            handle.call_tool("add", r#"{"x": 1, "y": 2}"#),
        )
        .await
        .expect("the call must not wait for the fetch to finish");

        assert_eq!(call.unwrap(), "3");
        assert_eq!(fetch.await.unwrap().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn a_failed_definitions_lookup_comes_back_as_an_error() {
        let handle = ToolServer::new()
            .dynamic_tools(1, FailingIndex, ToolSet::default())
            .tool(Adder)
            .run();

        let failed = handle.get_tool_defs(Some("query".into())).await;
        assert!(
            matches!(failed, Err(ToolServerError::DefinitionsFailed(_))),
            "{failed:?}"
        );
        let cause = std::error::Error::source(&failed.unwrap_err()).map(ToString::to_string);
        assert_eq!(
            cause.as_deref(),
            Some("RequestError: Datastore error: store is down")
        );

        let next = handle.call_tool("add", r#"{"x": 1, "y": 2}"#).await;
        assert_eq!(next.unwrap(), "3", "the handle keeps serving calls");
    }

    #[tokio::test]
    async fn a_panicking_definition_fails_only_its_fetch() {
        let handle = ToolServer::new().tool(PanickyDefinition).tool(Adder).run();

        let failed = handle.get_tool_defs(None).await.unwrap_err();
        assert_eq!(
            failed.to_string(),
            "Tool definitions failed: RequestError: tool definitions panicked: definition blew up"
        );

        let next = handle.call_tool("add", r#"{"x": 1, "y": 2}"#).await;
        assert_eq!(next.unwrap(), "3", "the handle keeps serving calls");
    }

    struct SlowDrop;

    impl Drop for SlowDrop {
        fn drop(&mut self) {
            std::thread::sleep(Duration::from_millis(300));
        }
    }

    impl Tool for SlowDrop {
        const NAME: &'static str = "slow_drop";
        type Error = MathError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            bare_definition(Self::NAME)
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("ok".into())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn removing_a_tool_does_not_hold_up_other_calls() {
        let handle = ToolServer::new().tool(SlowDrop).tool(Adder).run();
        let remover = tokio::spawn({
            let handle = handle.clone();
            async move { handle.remove_tool("slow_drop").await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;

        // A blocked lock stalls the thread itself, so a timeout around the
        // call would never fire. Measure the call instead.
        let start = std::time::Instant::now();
        let call = handle.call_tool("add", r#"{"x": 1, "y": 2}"#).await;
        let waited = start.elapsed();

        assert_eq!(call.unwrap(), "3");
        assert!(
            waited < Duration::from_millis(100),
            "the call waited {waited:?} for the removed tool to drop"
        );
        remover.await.unwrap().unwrap();
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
        let barrier = Arc::new(Barrier::new(2));
        let handle = ToolServer::new().tool(Meets(barrier)).run();
        let call_from_a_task = || {
            let handle = handle.clone();
            tokio::spawn(async move { handle.call_tool("meets", "{}").await })
        };
        let (first, second) = (call_from_a_task(), call_from_a_task());

        let both = tokio::time::timeout(Duration::from_secs(5), async {
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
            "Toolset error: ToolCallError: ToolCallError: Connection refused"
        );

        let missing = handle.call_tool("missing", "{}").await.unwrap_err();
        assert_eq!(
            missing.to_string(),
            "Toolset error: ToolNotFoundError: missing"
        );
    }

    #[tokio::test]
    async fn a_failed_call_keeps_its_error_type() {
        let handle = ToolServer::new().tool(Fails).run();

        let failed = handle.call_tool("fails", "{}").await.unwrap_err();
        let ToolServerError::ToolsetError(ToolSetError::ToolCallError(ToolError::ToolCallError(
            source,
        ))) = &failed
        else {
            panic!("expected the tool's own error, got {failed:?}");
        };
        assert!(source.downcast_ref::<Refused>().is_some());

        let missing = handle.call_tool("missing", "{}").await.unwrap_err();
        assert!(matches!(
            missing,
            ToolServerError::ToolsetError(ToolSetError::ToolNotFoundError(ref name)) if name == "missing"
        ));
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
