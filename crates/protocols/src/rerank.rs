use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::common::{default_true, GenerationRequest, StringOrArray, UsageInfo};

fn default_rerank_object() -> String {
    "rerank".to_string()
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs() as i64
}

// ============================================================================
// Rerank API
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_rerank_request"))]
pub struct RerankRequest {
    /// The query text to rank documents against
    #[validate(custom(function = "validate_query"))]
    pub query: String,

    /// List of documents to be ranked
    #[validate(custom(function = "validate_documents"))]
    pub documents: Vec<String>,

    /// Model to use for reranking
    pub model: String,

    /// Maximum number of documents to return (optional)
    #[serde(alias = "top_n", skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub top_k: Option<usize>,

    /// Whether to return documents in addition to scores
    #[serde(default = "default_true")]
    pub return_documents: bool,

    // SGLang specific extensions
    /// Request ID for tracking
    pub rid: Option<StringOrArray>,

    /// User identifier
    pub user: Option<String>,
}

impl GenerationRequest for RerankRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_ref().and_then(StringOrArray::first)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        false // Reranking doesn't support streaming
    }

    fn extract_text_for_routing(&self) -> String {
        self.query.clone()
    }
}

impl super::validated::Normalizable for RerankRequest {
    // Use default no-op normalization
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates that the query is not empty
fn validate_query(query: &str) -> Result<(), validator::ValidationError> {
    if query.trim().is_empty() {
        return Err(validator::ValidationError::new("query cannot be empty"));
    }
    Ok(())
}

/// Validates that the documents list is not empty
fn validate_documents(documents: &[String]) -> Result<(), validator::ValidationError> {
    if documents.is_empty() {
        return Err(validator::ValidationError::new(
            "documents list cannot be empty",
        ));
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
#[expect(
    clippy::unnecessary_wraps,
    reason = "validator crate requires Result return type"
)]
fn validate_rerank_request(req: &RerankRequest) -> Result<(), validator::ValidationError> {
    // Validate top_k if specified
    if let Some(k) = req.top_k {
        if k > req.documents.len() {
            tracing::warn!(
                "top_k ({}) is greater than number of documents ({})",
                k,
                req.documents.len()
            );
        }
    }
    Ok(())
}

impl RerankRequest {
    /// Get the effective top_k value
    pub fn effective_top_k(&self) -> usize {
        self.top_k.unwrap_or(self.documents.len())
    }
}

/// Individual rerank result
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResult {
    /// Relevance score for the document
    #[serde(alias = "relevance_score")]
    pub score: f32,

    /// The document text (if return_documents was true)
    #[serde(default, deserialize_with = "deserialize_rerank_document")]
    pub document: Option<String>,

    /// Original index of the document in the request
    pub index: usize,

    /// Additional metadata about the ranking
    pub meta_info: Option<HashMap<String, Value>>,
}

fn deserialize_rerank_document<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(Value::Object(mut document)) => document
            .remove("text")
            .and_then(|text| text.as_str().map(ToOwned::to_owned))
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("rerank document object must contain text")),
        Some(_) => Err(serde::de::Error::custom(
            "rerank document must be a string or object",
        )),
    }
}

/// Rerank response containing sorted results
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RerankResponse {
    /// Ranked results sorted by score (highest first)
    pub results: Vec<RerankResult>,

    /// Model used for reranking
    pub model: String,

    /// Usage information
    pub usage: Option<UsageInfo>,

    /// Response object type
    #[serde(default = "default_rerank_object")]
    pub object: String,

    /// Response ID
    pub id: Option<StringOrArray>,

    /// Creation timestamp
    pub created: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct V1RerankDocument {
    pub text: String,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct V1RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    pub document: Option<V1RerankDocument>,
    pub meta_info: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct V1RerankResponse {
    pub results: Vec<V1RerankResult>,
    pub model: String,
    pub usage: Option<UsageInfo>,
    pub object: String,
    pub id: Option<StringOrArray>,
    pub created: i64,
}

impl From<RerankResponse> for V1RerankResponse {
    fn from(response: RerankResponse) -> Self {
        Self {
            results: response
                .results
                .into_iter()
                .map(|result| V1RerankResult {
                    index: result.index,
                    relevance_score: result.score,
                    document: result.document.map(|text| V1RerankDocument { text }),
                    meta_info: result.meta_info,
                })
                .collect(),
            model: response.model,
            usage: response.usage,
            object: response.object,
            id: response.id,
            created: response.created,
        }
    }
}

impl RerankResponse {
    /// Create a new RerankResponse with the given results and model
    pub fn new(
        results: Vec<RerankResult>,
        model: String,
        request_id: Option<StringOrArray>,
    ) -> Self {
        RerankResponse {
            results,
            model,
            usage: None,
            object: default_rerank_object(),
            id: request_id,
            created: current_timestamp(),
        }
    }

    /// Apply top_k limit to results
    pub fn apply_top_k(&mut self, k: usize) {
        self.results.truncate(k);
    }

    /// Drop documents from results (when return_documents is false)
    pub fn drop_documents(&mut self) {
        for result in &mut self.results {
            result.document = None;
        }
    }
}

/// V1 API compatibility format for rerank requests
/// Matches Python's V1RerankReqInput
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct V1RerankReqInput {
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default, alias = "top_n")]
    pub top_k: Option<usize>,
    #[serde(default = "default_true")]
    pub return_documents: bool,
}

/// Convert V1RerankReqInput to RerankRequest
impl From<V1RerankReqInput> for RerankRequest {
    fn from(v1: V1RerankReqInput) -> Self {
        RerankRequest {
            query: v1.query,
            documents: v1.documents,
            model: v1
                .model
                .unwrap_or_else(|| super::UNKNOWN_MODEL_ID.to_string()),
            top_k: v1.top_k,
            return_documents: v1.return_documents,
            rid: None,
            user: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_response_uses_new_api_result_shape() {
        let response = RerankResponse::new(
            vec![RerankResult {
                score: 0.75,
                document: Some("document text".to_string()),
                index: 3,
                meta_info: None,
            }],
            "model".to_string(),
            None,
        );

        let value = serde_json::to_value(V1RerankResponse::from(response)).unwrap();
        assert_eq!(value["results"][0]["relevance_score"], 0.75);
        assert_eq!(value["results"][0]["document"]["text"], "document text");
        assert!(value["results"][0].get("score").is_none());
    }
}
