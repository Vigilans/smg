//! Multipart image editing through HTTP workers.

use axum::{body::Body, extract::Request, http::header::CONTENT_TYPE, response::Response};
use reqwest::multipart::Form;

use super::{convert_reqwest_error, multipart::MultipartUpload, Router};
use crate::{
    routers::{
        common::{
            header_utils, overload,
            placement::{self, PlacementFailure},
        },
        error,
    },
    worker::{RoutingPool, WorkerLoadGuard},
};

impl Router {
    pub(super) async fn handle_image_edit(&self, req: Request<Body>) -> Response {
        let headers = req.headers().clone();
        let upload = match MultipartUpload::parse(req).await {
            Ok(upload) => upload,
            Err(response) => return response,
        };
        let canonical = self.worker_registry.resolve_model_alias(&upload.model);
        let model = canonical.as_deref().unwrap_or(&upload.model);
        let worker = match self.select_worker_for_model(
            model,
            upload.prompt.as_deref(),
            None,
            Some(&headers),
            None,
            None,
        ) {
            Some(worker) => worker,
            None => {
                return match placement::single_failure(
                    &self.worker_registry,
                    model,
                    RoutingPool::HttpRegular,
                    None,
                ) {
                    PlacementFailure::NoCandidates => error::model_not_found(model),
                    PlacementFailure::AllOverloaded(response) => response,
                    PlacementFailure::Unavailable | PlacementFailure::PolicyDeclined(_) => {
                        error::service_unavailable(
                            "no_available_workers",
                            "No image workers are available",
                        )
                    }
                };
            }
        };
        if let Some(response) = overload::shed_if_worker_overloaded(worker.as_ref(), model) {
            return response;
        }
        let mut form = Form::new().text("model", model.to_owned());
        for (name, part) in upload.parts {
            form = form.part(name, part);
        }
        let builder = worker
            .http_client()
            .post(worker.endpoint_url("/v1/images/edits"))
            .multipart(form);
        let builder = header_utils::apply_forwarded_request_headers(
            builder,
            Some(&headers),
            worker.api_key(),
        );
        let load_guard = WorkerLoadGuard::with_key(worker.clone(), None);
        let upstream = match builder.send().await {
            Ok(response) => response,
            Err(error) => return convert_reqwest_error(error),
        };
        worker.record_outcome(upstream.status().as_u16());
        let is_stream = upstream
            .headers()
            .get(CONTENT_TYPE)
            .is_some_and(|value| value.as_bytes().starts_with(b"text/event-stream"));
        self.forward_worker_response(upstream, is_stream, worker.url(), load_guard)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{
        body::to_bytes,
        extract::Multipart,
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::post,
    };
    use bytes::Bytes;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
    use tokio::{net::TcpListener, sync::Mutex};

    use super::*;
    use crate::{config::types::PolicyConfig, worker::BasicWorkerBuilder};

    #[tokio::test]
    async fn image_edits_preserve_multipart_errors_and_streaming() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let capture = captured.clone();
        let app = axum::Router::new().route(
            "/v1/images/edits",
            post(move |headers: HeaderMap, mut form: Multipart| {
                let capture = capture.clone();
                async move {
                    assert_eq!(headers["authorization"], "Bearer client-key");
                    let mut parts = Vec::new();
                    while let Some(field) = form.next_field().await.unwrap() {
                        let name = field.name().unwrap().to_owned();
                        let filename = field.file_name().map(str::to_owned);
                        let mime = field.content_type().map(str::to_owned);
                        let bytes = field.bytes().await.unwrap();
                        parts.push((name, filename, mime, bytes));
                    }
                    let invalid = parts.iter().any(|p| p.0 == "true_cfg_scale" && p.3 == "21");
                    let stream = parts.iter().any(|p| p.0 == "stream" && p.3 == "true");
                    *capture.lock().await = parts;
                    if invalid {
                        return (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            [("content-type", "application/json")],
                            r#"{"detail":"true_cfg_scale must be <= 20"}"#,
                        )
                            .into_response();
                    }
                    if stream {
                        return (
                            [("content-type", "text/event-stream")],
                            "data: {\"b64_json\":\"cG5n\"}\n\ndata: [DONE]\n\n",
                        )
                            .into_response();
                    }
                    (
                        [("content-type", "application/json")],
                        r#"{"data":[{"b64_json":"cG5n"}]}"#,
                    )
                        .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "test server is aborted at test end"
        )]
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let worker = BasicWorkerBuilder::new(&url)
            .model(ModelCard::new("image-model").with_alias("image-alias"))
            .api_key("worker-key")
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        let router = super::super::tests::streaming_router(
            PolicyConfig::RoundRobin,
            1024 * 1024,
            vec![worker],
        );

        for (cfg, stream, status, content_type, expected_body) in [
            (
                "4",
                false,
                StatusCode::OK,
                "application/json",
                r#"{"data":[{"b64_json":"cG5n"}]}"#,
            ),
            (
                "21",
                false,
                StatusCode::UNPROCESSABLE_ENTITY,
                "application/json",
                r#"{"detail":"true_cfg_scale must be <= 20"}"#,
            ),
            (
                "4",
                true,
                StatusCode::OK,
                "text/event-stream",
                "data: {\"b64_json\":\"cG5n\"}\n\ndata: [DONE]\n\n",
            ),
        ] {
            let body = format!(concat!(
                "--edit-test\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"first.png\"\r\nContent-Type: image/png\r\n\r\nfirst\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nimage-alias\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nChange the boat color\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"true_cfg_scale\"\r\n\r\n{}\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\n{}\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"second.webp\"\r\nContent-Type: image/webp\r\n\r\nsecond\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"mask_image\"; filename=\"mask.png\"\r\nContent-Type: image/png\r\n\r\nmask\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"negative_prompt\"\r\n\r\n\r\n",
                "--edit-test\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n0\r\n",
                "--edit-test--\r\n"
            ), cfg, stream);
            let request = Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header("content-type", "multipart/form-data; boundary=edit-test")
                .header("authorization", "Bearer client-key")
                .body(Body::from(body))
                .unwrap();
            let response = router.handle_image_edit(request).await;
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()[CONTENT_TYPE], content_type);
            assert_eq!(
                to_bytes(response.into_body(), 4096).await.unwrap(),
                expected_body
            );

            let parts = captured.lock().await;
            assert_eq!(
                parts.iter().find(|p| p.0 == "model").unwrap().3,
                "image-model"
            );
            assert_eq!(
                parts.iter().find(|p| p.0 == "true_cfg_scale").unwrap().3,
                cfg
            );
            assert_eq!(
                parts.iter().find(|p| p.0 == "negative_prompt").unwrap().3,
                Bytes::new()
            );
            assert_eq!(parts.iter().find(|p| p.0 == "seed").unwrap().3, "0");
            let media: Vec<_> = parts.iter().filter(|p| p.1.is_some()).collect();
            assert_eq!(media.len(), 3);
            assert_eq!(media[0].0, "image[]");
            assert_eq!(media[0].1.as_deref(), Some("first.png"));
            assert_eq!(media[0].2.as_deref(), Some("image/png"));
            assert_eq!(media[0].3, "first");
            assert_eq!(media[1].0, "image[]");
            assert_eq!(media[1].1.as_deref(), Some("second.webp"));
            assert_eq!(media[1].2.as_deref(), Some("image/webp"));
            assert_eq!(media[1].3, "second");
            assert_eq!(media[2].0, "mask_image");
            assert_eq!(media[2].3, "mask");
        }
        server.abort();
    }
}
