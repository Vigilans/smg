//! vLLM-native `/v1/decisions` and `/v1/systemone` over gRPC.
//!
//! The JSON body goes to the vLLM worker unchanged; the worker runs vLLM's own
//! handler on its engine and returns the HTTP status and JSON body that the
//! worker's HTTP route would return.

use std::sync::Arc;

use axum::{
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::Value;
use smg_grpc_client::vllm_proto::{NativeApiRequest, NativeApiResponse};
use tracing::warn;

use super::backend_client::BackendClient;
use crate::{
    policies::PolicyRegistry,
    routers::{
        common::{
            overload,
            placement::{self, PlacementFailure, PlacementInputs},
        },
        error, NativeDecisionApi,
    },
    worker::{ConnectionMode, RoutingPool, RuntimeType, Worker, WorkerLoadGuard, WorkerRegistry},
};

/// Worker label listing the native decision APIs a vLLM worker serves.
const NATIVE_DECISION_APIS_LABEL: &str = "native_decision_apis";

/// Request text for cache-aware placement; the engine renders the real prompt.
fn routing_text(api: NativeDecisionApi, body: &Value) -> Option<String> {
    let value = match api {
        NativeDecisionApi::Decisions => body.get("input")?,
        NativeDecisionApi::SystemOne => body.get("state")?,
    };
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    if api == NativeDecisionApi::Decisions {
        let text = value
            .as_array()?
            .iter()
            .filter_map(|message| message.get("content"))
            .flat_map(|content| match content {
                Value::String(text) => vec![text.as_str()],
                Value::Array(parts) => parts
                    .iter()
                    .filter(|part| part.get("type").and_then(Value::as_str) == Some("input_text"))
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect(),
                _ => Vec::new(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Some(text);
    }
    serde_json::to_string(value).ok()
}

fn serves(worker: &dyn Worker, api: NativeDecisionApi) -> bool {
    let spec = &worker.metadata().spec;
    spec.runtime_type == RuntimeType::Vllm
        && *worker.connection_mode() == ConnectionMode::Grpc
        && spec
            .labels
            .get(NATIVE_DECISION_APIS_LABEL)
            .is_some_and(|apis| apis.split(',').any(|name| name == api.label()))
}

fn candidates(
    registry: &WorkerRegistry,
    model_id: &str,
    api: NativeDecisionApi,
) -> Vec<Arc<dyn Worker>> {
    registry
        .get_routing_pool(model_id, RoutingPool::GrpcPipelineRegular)
        .iter()
        .filter(|worker| serves(worker.as_ref(), api))
        .cloned()
        .collect()
}

fn into_response(response: NativeApiResponse) -> Response {
    let status = u16::try_from(response.status_code)
        .ok()
        .and_then(|code| StatusCode::from_u16(code).ok())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    (
        status,
        [(CONTENT_TYPE, HeaderValue::from_static("application/json"))],
        response.response_json,
    )
        .into_response()
}

async fn call(
    worker: &Arc<dyn Worker>,
    api: NativeDecisionApi,
    request: NativeApiRequest,
) -> Result<NativeApiResponse, String> {
    let client = worker
        .get_backend_client()
        .await
        .map_err(|e| e.to_string())?
        .ok_or("worker has no gRPC client")?;
    let BackendClient::Grpc(client) = client.as_ref() else {
        return Err("worker is not a gRPC worker".to_string());
    };
    let client = client.as_vllm();
    match api {
        NativeDecisionApi::Decisions => client.decisions(request).await,
        NativeDecisionApi::SystemOne => client.system_one(request).await,
    }
    .map_err(|status| status.to_string())
}

/// Select a capable vLLM gRPC worker and return its native handler's result.
/// A transport failure retries once on another candidate; engine responses,
/// including errors, return as they are.
pub(crate) async fn route(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    headers: Option<&HeaderMap>,
    api: NativeDecisionApi,
    mut body: Value,
    model_id: &str,
) -> Response {
    if let Some(object) = body.as_object_mut() {
        object.insert("model".to_string(), Value::String(model_id.to_string()));
    }
    let text = routing_text(api, &body);
    let request = NativeApiRequest {
        request_json: body.to_string(),
    };
    let mut candidates = candidates(registry, model_id, api);
    for attempt in 0..2 {
        let Some(worker) = placement::select_from(
            registry,
            policies,
            model_id,
            &candidates,
            PlacementInputs {
                text: text.as_deref(),
                tokens: None,
                headers,
                rid_key: None,
                cache_namespace: None,
            },
        ) else {
            return match placement::failure_from(&candidates, model_id) {
                PlacementFailure::AllOverloaded(shed) => shed,
                _ => error::service_unavailable(
                    "native_decision_unavailable",
                    format!(
                        "No vLLM gRPC worker serves {} for model '{model_id}'",
                        api.label()
                    ),
                ),
            };
        };
        if let Some(shed) = overload::shed_if_worker_overloaded(worker.as_ref(), model_id) {
            return shed;
        }
        let _guard = WorkerLoadGuard::new(worker.clone(), headers);
        match call(&worker, api, request.clone()).await {
            Ok(response) => {
                worker.record_outcome(response.status_code.try_into().unwrap_or(u16::MAX));
                return into_response(response);
            }
            Err(message) => {
                worker.record_outcome(StatusCode::BAD_GATEWAY.as_u16());
                warn!(
                    worker = worker.url(),
                    attempt, "Native decision RPC failed: {message}"
                );
                if attempt == 1 {
                    return error::bad_gateway("native_decision_failed", message);
                }
                candidates.retain(|candidate| !Arc::ptr_eq(candidate, &worker));
            }
        }
    }
    unreachable!("the second attempt always returns")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::worker::HealthCheckConfig;
    use serde_json::json;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        worker::{BasicWorkerBuilder, ModelCard, WorkerType},
    };

    #[test]
    fn routing_text_uses_request_content() {
        assert_eq!(
            routing_text(NativeDecisionApi::Decisions, &json!({"input": "plain"})).as_deref(),
            Some("plain")
        );
        let messages = json!({"input": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": [
                {"type": "input_text", "text": "second"},
                {"type": "input_image", "image_url": "https://example.com/a.png"},
                {"type": "input_text", "text": "third"}
            ]}
        ]});
        assert_eq!(
            routing_text(NativeDecisionApi::Decisions, &messages).as_deref(),
            Some("first\nsecond\nthird")
        );
        assert_eq!(
            routing_text(NativeDecisionApi::SystemOne, &json!({"state": "ticket"})).as_deref(),
            Some("ticket")
        );
        assert_eq!(
            routing_text(NativeDecisionApi::SystemOne, &json!({"state": {"a": 1}})).as_deref(),
            Some(r#"{"a":1}"#)
        );
    }

    fn worker(url: &str, runtime: RuntimeType, apis: Option<&str>) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new("model-x"))
            .worker_type(WorkerType::Regular)
            .runtime_type(runtime)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        if let Some(apis) = apis {
            builder = builder.labels(HashMap::from([(
                NATIVE_DECISION_APIS_LABEL.to_string(),
                apis.to_string(),
            )]));
        }
        Arc::new(builder.build())
    }

    #[test]
    fn candidates_require_vllm_capability() {
        let registry = WorkerRegistry::new();
        for worker in [
            worker(
                "grpc://127.0.0.1:9500",
                RuntimeType::Vllm,
                Some("decisions,systemone"),
            ),
            worker(
                "grpc://127.0.0.1:9501",
                RuntimeType::Vllm,
                Some("decisions"),
            ),
            worker("grpc://127.0.0.1:9502", RuntimeType::Vllm, None),
            worker(
                "grpc://127.0.0.1:9503",
                RuntimeType::Sglang,
                Some("decisions,systemone"),
            ),
        ] {
            registry.register(worker).unwrap();
        }
        let urls = |api| {
            let mut urls: Vec<_> = candidates(&registry, "model-x", api)
                .iter()
                .map(|worker| worker.url().to_string())
                .collect();
            urls.sort();
            urls
        };
        assert_eq!(
            urls(NativeDecisionApi::Decisions),
            ["grpc://127.0.0.1:9500", "grpc://127.0.0.1:9501"]
        );
        assert_eq!(
            urls(NativeDecisionApi::SystemOne),
            ["grpc://127.0.0.1:9500"]
        );
    }

    #[tokio::test]
    async fn no_capable_worker_is_unavailable() {
        let registry = WorkerRegistry::new();
        registry
            .register(worker(
                "grpc://127.0.0.1:9504",
                RuntimeType::Sglang,
                Some("decisions"),
            ))
            .unwrap();
        let response = route(
            &registry,
            &PolicyRegistry::new(PolicyConfig::RoundRobin),
            None,
            NativeDecisionApi::Decisions,
            json!({"model": "model-x", "input": "x"}),
            "model-x",
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "native_decision_unavailable"
        );
    }

    #[tokio::test]
    async fn transport_failure_retries_once_on_another_worker() {
        let registry = WorkerRegistry::new();
        let workers = [
            worker("grpc://127.0.0.1:1", RuntimeType::Vllm, Some("decisions")),
            worker("grpc://127.0.0.1:2", RuntimeType::Vllm, Some("decisions")),
        ];
        for worker in &workers {
            registry.register(worker.clone()).unwrap();
        }
        let response = route(
            &registry,
            &PolicyRegistry::new(PolicyConfig::RoundRobin),
            None,
            NativeDecisionApi::Decisions,
            json!({"model": "model-x", "input": "x"}),
            "model-x",
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "native_decision_failed"
        );
        for worker in &workers {
            assert_eq!(worker.load(), 0);
        }
    }

    #[tokio::test]
    async fn engine_status_and_body_pass_through() {
        for (code, status) in [
            (200, StatusCode::OK),
            (400, StatusCode::BAD_REQUEST),
            (501, StatusCode::NOT_IMPLEMENTED),
        ] {
            let response = into_response(NativeApiResponse {
                status_code: code,
                response_json: format!(r#"{{"code":{code}}}"#),
            });
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(body, format!(r#"{{"code":{code}}}"#));
        }
    }
}
