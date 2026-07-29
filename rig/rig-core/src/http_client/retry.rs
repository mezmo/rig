//! Helpers to handle connection delays when receiving errors

use super::Error;
use std::time::Duration;

pub trait RetryPolicy {
    /// Submit a new retry delay based on the [`enum@Error`], last retry number and duration, if
    /// available. A policy may also return `None` if it does not want to retry
    fn retry(&self, error: &Error, last_retry: Option<(usize, Duration)>) -> Option<Duration>;

    /// Set a new reconnection time if received from an event
    fn set_reconnection_time(&mut self, duration: Duration);
}

/// A [`RetryPolicy`] which backs off exponentially
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    /// The start of the backoff
    pub start: Duration,
    /// The factor of which to backoff by
    pub factor: f64,
    /// The maximum duration to delay
    pub max_duration: Option<Duration>,
    /// The maximum number of retries before giving up
    pub max_retries: Option<usize>,
}

impl ExponentialBackoff {
    /// Create a new exponential backoff retry policy
    pub const fn new(
        start: Duration,
        factor: f64,
        max_duration: Option<Duration>,
        max_retries: Option<usize>,
    ) -> Self {
        Self {
            start,
            factor,
            max_duration,
            max_retries,
        }
    }
}

impl RetryPolicy for ExponentialBackoff {
    fn retry(&self, _error: &Error, last_retry: Option<(usize, Duration)>) -> Option<Duration> {
        if let Some((retry_num, last_duration)) = last_retry {
            if self.max_retries.is_none() || retry_num < self.max_retries.unwrap() {
                let duration = last_duration.mul_f64(self.factor);
                if let Some(max_duration) = self.max_duration {
                    Some(duration.min(max_duration))
                } else {
                    Some(duration)
                }
            } else {
                None
            }
        } else {
            Some(self.start)
        }
    }
    fn set_reconnection_time(&mut self, duration: Duration) {
        self.start = duration;
        if let Some(max_duration) = self.max_duration {
            self.max_duration = Some(max_duration.max(duration))
        }
    }
}

/// A [`RetryPolicy`] which always emits the same delay
#[derive(Debug, Clone)]
pub struct Constant {
    /// The delay to return
    pub delay: Duration,
    /// The maximum number of retries to return before giving up
    pub max_retries: Option<usize>,
}

impl Constant {
    /// Create a new constant retry policy
    pub const fn new(delay: Duration, max_retries: Option<usize>) -> Self {
        Self { delay, max_retries }
    }
}

impl RetryPolicy for Constant {
    fn retry(&self, _error: &Error, last_retry: Option<(usize, Duration)>) -> Option<Duration> {
        if let Some((retry_num, _)) = last_retry {
            if self.max_retries.is_none() || retry_num < self.max_retries.unwrap() {
                Some(self.delay)
            } else {
                None
            }
        } else {
            Some(self.delay)
        }
    }
    fn set_reconnection_time(&mut self, duration: Duration) {
        self.delay = duration;
    }
}

/// A [`RetryPolicy`] which never retries
#[derive(Debug, Clone, Copy, Default)]
pub struct Never;

impl RetryPolicy for Never {
    fn retry(&self, _error: &Error, _last_retry: Option<(usize, Duration)>) -> Option<Duration> {
        None
    }
    fn set_reconnection_time(&mut self, _duration: Duration) {}
}

/// The default [`RetryPolicy`] when initializing an event source
pub const DEFAULT_RETRY: ExponentialBackoff = ExponentialBackoff::new(
    Duration::from_millis(300),
    2.,
    Some(Duration::from_secs(5)),
    None,
);

/// A [`RetryPolicy`] for retryable HTTP status responses (429, 500, 502, 503, 504, 529).
#[derive(Debug, Clone)]
pub struct StatusRetry {
    /// The initial backoff delay.
    pub start: Duration,
    /// The factor by which the backoff grows between attempts.
    pub factor: f64,
    /// The upper bound on any single delay.
    pub max_duration: Duration,
    /// The maximum number of retry attempts before giving up.
    pub max_retries: usize,
}

impl RetryPolicy for StatusRetry {
    fn retry(&self, error: &Error, last_retry: Option<(usize, Duration)>) -> Option<Duration> {
        let (status, retry_after) = match error {
            Error::InvalidStatusCode(status) => (*status, None),
            Error::InvalidStatusCodeWithMessage(status, _, retry_after) => (*status, *retry_after),
            _ => return None,
        };
        if !is_retryable_status(status) {
            return None;
        }
        let retry_num = last_retry.map(|(n, _)| n).unwrap_or(0);
        if retry_num >= self.max_retries {
            return None;
        }
        let delay = match retry_after {
            Some(hint) => hint.min(self.max_duration),
            None => match last_retry {
                Some((_, last_duration)) => {
                    last_duration.mul_f64(self.factor).min(self.max_duration)
                }
                None => self.start,
            },
        };
        Some(delay)
    }

    fn set_reconnection_time(&mut self, _duration: Duration) {}
}

fn is_retryable_status(status: http::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504 | 529)
}

/// The default [`StatusRetry`] used at stream open.
pub const DEFAULT_STATUS_RETRY: StatusRetry = StatusRetry {
    start: Duration::from_secs(1),
    factor: 2.0,
    max_duration: Duration::from_secs(16),
    max_retries: 5,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_client::instance_error;
    use http::StatusCode;

    fn status_err(status: u16, retry_after: Option<Duration>) -> Error {
        Error::InvalidStatusCodeWithMessage(
            StatusCode::from_u16(status).unwrap(),
            "body".to_string(),
            retry_after,
        )
    }

    fn policy() -> StatusRetry {
        StatusRetry {
            start: Duration::from_millis(100),
            factor: 2.0,
            max_duration: Duration::from_secs(10),
            max_retries: 5,
        }
    }

    #[test]
    fn retryable_status_returns_start() {
        let p = policy();
        assert_eq!(
            p.retry(&status_err(429, None), None),
            Some(Duration::from_millis(100))
        );
    }

    #[test]
    fn honors_retry_after_hint() {
        let p = policy();
        assert_eq!(
            p.retry(&status_err(429, Some(Duration::from_secs(3))), None),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn caps_retry_after_at_max_duration() {
        let p = policy();
        assert_eq!(
            p.retry(&status_err(429, Some(Duration::from_secs(30))), None),
            Some(Duration::from_secs(10))
        );
    }

    #[test]
    fn non_retryable_status_returns_none() {
        let p = policy();
        assert_eq!(p.retry(&status_err(404, None), None), None);
    }

    #[test]
    fn bare_invalid_status_code_retries_without_hint() {
        let p = policy();
        assert_eq!(
            p.retry(
                &Error::InvalidStatusCode(StatusCode::from_u16(429).unwrap()),
                None
            ),
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            p.retry(
                &Error::InvalidStatusCode(StatusCode::from_u16(404).unwrap()),
                None
            ),
            None
        );
    }

    #[test]
    fn transport_error_returns_none() {
        let p = policy();
        let err = instance_error(std::io::Error::other("boom"));
        assert_eq!(p.retry(&err, None), None);
    }

    #[test]
    fn exhausts_at_max_retries() {
        let p = StatusRetry {
            start: Duration::from_millis(1),
            factor: 2.0,
            max_duration: Duration::from_millis(4),
            max_retries: 3,
        };
        let mut last = None;
        let mut delays = Vec::new();
        while let Some(delay) = p.retry(&status_err(429, None), last) {
            let next_num = last.map(|(n, _)| n + 1).unwrap_or(1);
            last = Some((next_num, delay));
            delays.push(delay);
        }
        assert_eq!(delays.len(), 3);
        assert_eq!(delays[0], Duration::from_millis(1));
        assert_eq!(delays[1], Duration::from_millis(2));
        assert_eq!(delays[2], Duration::from_millis(4));
        assert_eq!(p.retry(&status_err(429, None), last), None);
    }

    #[test]
    fn max_retries_zero_declines_immediately() {
        let p = StatusRetry {
            start: Duration::from_millis(100),
            factor: 2.0,
            max_duration: Duration::from_secs(10),
            max_retries: 0,
        };
        assert_eq!(p.retry(&status_err(429, None), None), None);
    }
}
