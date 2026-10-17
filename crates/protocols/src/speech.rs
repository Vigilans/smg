//! Audio speech generation API protocol definitions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::GenerationRequest;

/// OpenAI-compatible `/v1/audio/speech` request.
///
/// Backend-specific fields are retained verbatim so HTTP workers can extend
/// the API without requiring gateway protocol changes.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct SpeechRequest {
    pub model: String,

    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl GenerationRequest for SpeechRequest {
    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        self.extra
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || self
                .extra
                .get("stream_format")
                .and_then(Value::as_str)
                .is_some_and(|format| format.eq_ignore_ascii_case("sse"))
    }

    fn extract_text_for_routing(&self) -> String {
        self.extra
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_backend_extensions() {
        let request: SpeechRequest = serde_json::from_value(serde_json::json!({
            "model": "tts",
            "input": "hello",
            "voice": "default",
            "ref_audio": "data:audio/wav;base64,AAAA",
            "stream": true
        }))
        .unwrap();

        assert!(request.is_stream());
        assert_eq!(request.extract_text_for_routing(), "hello");
        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["ref_audio"], "data:audio/wav;base64,AAAA");
    }

    #[test]
    fn stream_format_sse_enables_streaming() {
        let request: SpeechRequest = serde_json::from_value(serde_json::json!({
            "model": "tts",
            "input": "hello",
            "stream_format": "sse"
        }))
        .unwrap();

        assert!(request.is_stream());
    }
}
