//! Image generation API protocol definitions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::GenerationRequest;

/// OpenAI-compatible `/v1/images/generations` request.
///
/// Sampling parameters and backend extensions are forwarded verbatim.
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,

    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl GenerationRequest for ImageGenerationRequest {
    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        self.extra
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    fn extract_text_for_routing(&self) -> String {
        self.prompt.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_image_parameters_and_backend_extensions() {
        let body = serde_json::json!({
            "model": "image-model",
            "prompt": "A lighthouse at sunset",
            "size": "2048x1152",
            "n": 1,
            "response_format": "b64_json",
            "seed": 42,
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "negative_prompt": "blurry",
            "extra_params": {"custom_setting": [1, 2]},
            "stream": true
        });
        let request: ImageGenerationRequest = serde_json::from_value(body.clone()).unwrap();

        assert_eq!(request.get_model(), Some("image-model"));
        assert_eq!(request.extract_text_for_routing(), "A lighthouse at sunset");
        assert!(request.is_stream());
        assert_eq!(serde_json::to_value(request).unwrap(), body);
    }
}
