//! Form video submission and worker affinity for asynchronous jobs.

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{
        header::{CONTENT_TYPE, IF_RANGE, RANGE},
        HeaderMap, Method, StatusCode,
    },
    response::{IntoResponse, Response},
};
use futures_util::{stream, StreamExt};
use reqwest::multipart::Form;
use serde_json::Value;

use super::{convert_reqwest_error, multipart::MultipartUpload, Router};
use crate::{
    routers::{
        common::{
            header_utils, overload,
            placement::{self, PlacementFailure},
        },
        error,
    },
    worker::{RoutingPool, Worker, WorkerLoadGuard, UNKNOWN_MODEL_ID},
};

impl Router {
    pub(super) async fn handle_video_request(&self, req: Request<Body>) -> Response {
        if req.method() == Method::POST {
            return self.create_video(req).await;
        }

        let path = req.uri().path();
        let Some(resource) = path.strip_prefix("/v1/videos/") else {
            return error::not_found("video_not_found", "Video not found");
        };
        let (video_id, content) = match resource.strip_suffix("/content") {
            Some(id) => (id, true),
            None => (resource, false),
        };
        let worker = match self.find_video_worker(video_id, req.headers()).await {
            Ok(worker) => worker,
            Err(response) => return response,
        };
        let head = req.method() == Method::HEAD;
        // Omni exposes downloads through GET. HEAD obtains those headers and
        // closes the upstream response without downloading its body.
        let method = if head {
            Method::GET
        } else {
            req.method().clone()
        };
        let endpoint = req.uri().path_and_query().map_or(path, |p| p.as_str());
        let builder = worker
            .http_client()
            .request(method.clone(), worker.endpoint_url(endpoint));
        let mut builder = header_utils::apply_forwarded_request_headers(
            builder,
            Some(req.headers()),
            worker.api_key(),
        );
        if content {
            for name in [RANGE, IF_RANGE] {
                if let Some(value) = req.headers().get(&name) {
                    builder = builder.header(name, value);
                }
            }
        }
        let load_guard = WorkerLoadGuard::with_key(worker.clone(), None);
        let upstream = match builder.send().await {
            Ok(response) => response,
            Err(error) => return convert_reqwest_error(error),
        };
        if (method == Method::DELETE && upstream.status().is_success())
            || (!content && upstream.status() == StatusCode::NOT_FOUND)
        {
            self.video_workers.remove(video_id);
        }
        if head {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = upstream.status();
            *response.headers_mut() = header_utils::preserve_response_headers(upstream.headers());
            header_utils::insert_routed_worker_id(response.headers_mut(), worker.url());
            return response;
        }
        self.forward_worker_response(upstream, content, worker.url(), load_guard)
            .await
    }

    async fn create_video(&self, req: Request<Body>) -> Response {
        let headers = req.headers().clone();
        let urlencoded = headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .is_some_and(|value| {
                value
                    .trim()
                    .eq_ignore_ascii_case("application/x-www-form-urlencoded")
            });
        let upload = if urlencoded {
            MultipartUpload::parse_urlencoded(req).await
        } else {
            MultipartUpload::parse(req).await
        };
        let upload = match upload {
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
                            "No video workers are available",
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
            .post(worker.endpoint_url("/v1/videos"))
            .multipart(form);
        let builder = header_utils::apply_forwarded_request_headers(
            builder,
            Some(&headers),
            worker.api_key(),
        );
        let load_guard = WorkerLoadGuard::with_key(worker.clone(), None);
        // Submission creates a new job, so an ambiguous transport failure
        // must be returned to the caller without resubmitting it.
        let upstream = match builder.send().await {
            Ok(response) => response,
            Err(error) => return convert_reqwest_error(error),
        };
        worker.record_outcome(upstream.status().as_u16());
        let response = self
            .forward_worker_response(upstream, false, worker.url(), load_guard)
            .await;
        if !response.status().is_success() {
            return response;
        }
        let (parts, body) = response.into_parts();
        let bytes = match to_bytes(body, self.max_payload_size).await {
            Ok(bytes) => bytes,
            Err(error) => return error::bad_gateway("video_response_invalid", error.to_string()),
        };
        let metadata = serde_json::from_slice::<Value>(&bytes).ok();
        let Some(id) = metadata
            .as_ref()
            .and_then(|v| v.get("id"))
            .and_then(Value::as_str)
        else {
            return error::bad_gateway("video_response_invalid", "Video response has no job ID");
        };
        self.video_workers
            .insert(id.to_owned(), worker.url().to_owned());
        Response::from_parts(parts, Body::from(bytes))
    }

    async fn find_video_worker(
        &self,
        video_id: &str,
        headers: &HeaderMap,
    ) -> Result<Arc<dyn Worker>, Response> {
        if let Some(url) = self
            .video_workers
            .get(video_id)
            .map(|entry| entry.value().clone())
        {
            return self.worker_registry.get_by_url(&url).ok_or_else(|| {
                error::service_unavailable(
                    "video_worker_unavailable",
                    "The video's worker is unavailable",
                )
            });
        }

        // Jobs survive gateway restarts in the backend. Recover affinity with
        // read-only metadata lookups before forwarding a job operation.
        let workers = self
            .worker_registry
            .get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
        let lookups: Vec<_> = workers
            .iter()
            .cloned()
            .map(|worker: Arc<dyn Worker>| async move {
                let builder = worker
                    .http_client()
                    .get(worker.endpoint_url(&format!("/v1/videos/{video_id}")))
                    .timeout(Duration::from_secs(10));
                let builder = header_utils::apply_forwarded_request_headers(
                    builder,
                    Some(headers),
                    worker.api_key(),
                );
                let response = builder.send().await.map_err(convert_reqwest_error)?;
                let status = response.status();
                if matches!(
                    status,
                    StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED
                ) {
                    return Ok(None);
                }
                let bytes =
                    Self::read_worker_body_capped(response.bytes_stream(), self.max_payload_size)
                        .await?;
                let metadata = serde_json::from_slice::<Value>(&bytes).ok();
                if metadata
                    .as_ref()
                    .and_then(|v| v.get("id"))
                    .and_then(Value::as_str)
                    == Some(video_id)
                {
                    return Ok(Some(worker));
                }
                if status.is_server_error()
                    || status == StatusCode::UNAUTHORIZED
                    || status == StatusCode::FORBIDDEN
                {
                    return Err((status, Body::from(bytes)).into_response());
                }
                Ok(None)
            })
            .collect();
        let mut lookups = stream::iter(lookups).buffer_unordered(8);
        let mut failure = None;
        while let Some(result) = lookups.next().await {
            match result {
                Ok(Some(worker)) => {
                    self.video_workers
                        .insert(video_id.to_owned(), worker.url().to_owned());
                    return Ok(worker);
                }
                Ok(None) => {}
                Err(response) => failure = Some(response),
            }
        }
        Err(failure.unwrap_or_else(|| error::not_found("video_not_found", "Video not found")))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use axum::{
        extract::{Multipart, Path},
        routing::{get, post},
        Json,
    };
    use bytes::Bytes;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
    use serde_json::json;
    use tokio::{net::TcpListener, sync::Mutex, task::JoinHandle};

    use super::*;
    use crate::{config::types::PolicyConfig, worker::BasicWorkerBuilder};

    type CapturedParts = Arc<Mutex<Vec<(String, Option<String>, Option<String>, Bytes)>>>;

    async fn video_backend(id: &'static str) -> (String, CapturedParts, JoinHandle<()>) {
        let captured: CapturedParts = Arc::new(Mutex::new(Vec::new()));
        let capture = captured.clone();
        let deleted = Arc::new(AtomicBool::new(false));
        let get_deleted = deleted.clone();
        let app = axum::Router::new()
            .route(
                "/v1/videos",
                post(move |mut form: Multipart| {
                    let capture = capture.clone();
                    async move {
                        while let Some(field) = form.next_field().await.unwrap() {
                            let name = field.name().unwrap().to_owned();
                            let filename = field.file_name().map(str::to_owned);
                            let mime = field.content_type().map(str::to_owned);
                            let bytes = field.bytes().await.unwrap();
                            capture.lock().await.push((name, filename, mime, bytes));
                        }
                        Json(json!({"id": id, "model": "video-model", "status": "queued"}))
                    }
                }),
            )
            .route(
                "/v1/videos/{id}",
                get(move |Path(requested): Path<String>| {
                    let deleted = get_deleted.clone();
                    async move {
                        if requested != id || deleted.load(Ordering::SeqCst) {
                            return StatusCode::NOT_FOUND.into_response();
                        }
                        Json(json!({"id": id, "status": "completed", "progress": 100}))
                            .into_response()
                    }
                })
                .delete(move |Path(requested): Path<String>| {
                    let deleted = deleted.clone();
                    async move {
                        if requested != id {
                            return StatusCode::NOT_FOUND.into_response();
                        }
                        deleted.store(true, Ordering::SeqCst);
                        Json(json!({"id": id, "deleted": true})).into_response()
                    }
                }),
            )
            .route(
                "/v1/videos/{id}/content",
                get(
                    move |Path(requested): Path<String>, headers: HeaderMap| async move {
                        if requested != id {
                            return StatusCode::NOT_FOUND.into_response();
                        }
                        if headers.get(RANGE).is_some_and(|v| v == "bytes=0-1") {
                            return (
                                StatusCode::PARTIAL_CONTENT,
                                [
                                    ("content-type", "video/mp4"),
                                    ("content-range", "bytes 0-1/4"),
                                ],
                                "vi",
                            )
                                .into_response();
                        }
                        (
                            [("content-type", "video/mp4"), ("content-length", "4")],
                            "vide",
                        )
                            .into_response()
                    },
                ),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "test server is aborted at test end"
        )]
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, captured, server)
    }

    fn upload() -> Request<Body> {
        Request::builder().method(Method::POST).uri("/v1/videos")
            .header("content-type", "multipart/form-data; boundary=video-test")
            .body(Body::from(concat!(
                "--video-test\r\nContent-Disposition: form-data; name=\"input_references\"; filename=\"first.png\"\r\nContent-Type: image/png\r\n\r\nfirst\r\n",
                "--video-test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nvideo-alias\r\n",
                "--video-test\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nMoving water\r\n",
                "--video-test\r\nContent-Disposition: form-data; name=\"extra_params\"\r\n\r\n{\"task\":\"fl2va\",\"duration\":4}\r\n",
                "--video-test\r\nContent-Disposition: form-data; name=\"input_references\"; filename=\"last.png\"\r\nContent-Type: image/png\r\n\r\nlast\r\n",
                "--video-test--\r\n"
            ))).unwrap()
    }

    fn job_request(method: Method, id: &str, content: bool) -> Request<Body> {
        let suffix = if content { "/content" } else { "" };
        Request::builder()
            .method(method)
            .uri(format!("/v1/videos/{id}{suffix}"))
            .body(Body::empty())
            .unwrap()
    }

    async fn json_body(response: Response) -> Value {
        assert_eq!(response.status(), StatusCode::OK);
        serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn video_urlencoded_preserves_fields_and_routes_model_alias() {
        let (url, captured, server) = video_backend("video_gen_form").await;
        let worker = BasicWorkerBuilder::new(&url)
            .model(ModelCard::new("video-model").with_alias("video-alias"))
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
        let fields = [
            ("aspect_ratio", "16:9"),
            ("seconds", "4"),
            ("extra_params", r#"{"task":"t2va","duration":4}"#),
            (
                "image_reference",
                r#"{"image_url":"https://example.com/ref.png?token=a+b&x=1"}"#,
            ),
            ("negative_prompt", ""),
            ("seed", "0"),
            ("user", "first"),
            ("user", "last"),
        ];
        let body = format!(
            "model=video-alias&prompt=%E6%B5%B7%E8%BE%B9+waves%2Bwind+%26+light%3Dgold&{}",
            url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs(fields)
                .finish()
        );
        for content_type in [
            "application/x-www-form-urlencoded",
            "application/x-www-form-urlencoded; charset=UTF-8",
        ] {
            captured.lock().await.clear();
            let request = Request::builder()
                .method(Method::POST)
                .uri("/v1/videos")
                .header(CONTENT_TYPE, content_type)
                .body(Body::from(body.clone()))
                .unwrap();
            let response = json_body(router.handle_video_request(request).await).await;
            assert_eq!(response["id"], "video_gen_form");
            let parts = captured.lock().await;
            let actual: Vec<_> = parts
                .iter()
                .map(|part| (part.0.as_str(), std::str::from_utf8(&part.3).unwrap()))
                .collect();
            let expected: Vec<_> = [
                ("model", "video-model"),
                ("prompt", "海边 waves+wind & light=gold"),
            ]
            .into_iter()
            .chain(fields)
            .collect();
            assert_eq!(actual, expected);
            assert!(parts.iter().all(|part| part.1.is_none()));
            assert!(router.video_workers.contains_key("video_gen_form"));
        }
        server.abort();
    }

    #[tokio::test]
    async fn video_urlencoded_requires_model() {
        let router =
            super::super::tests::streaming_router(PolicyConfig::RoundRobin, 1024 * 1024, vec![]);
        for body in ["prompt=waves", "model=++&prompt=waves"] {
            let request = Request::builder()
                .method(Method::POST)
                .uri("/v1/videos")
                .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from(body))
                .unwrap();
            let response = router.handle_video_request(request).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            assert!(std::str::from_utf8(&body)
                .unwrap()
                .contains("missing_model"));
        }
    }

    #[tokio::test]
    async fn video_upload_affinity_recovery_download_and_delete() {
        let (url_a, captured_a, server_a) = video_backend("video_gen_a").await;
        let (url_b, captured_b, server_b) = video_backend("video_gen_b").await;
        let workers = [&url_a, &url_b]
            .into_iter()
            .map(|url| {
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("video-model").with_alias("video-alias"))
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build()
            })
            .collect();
        let router =
            super::super::tests::streaming_router(PolicyConfig::RoundRobin, 1024 * 1024, workers);
        let first = json_body(router.handle_video_request(upload()).await).await;
        let second = json_body(router.handle_video_request(upload()).await).await;
        let first_id = first["id"].as_str().unwrap();
        let second_id = second["id"].as_str().unwrap();
        assert_ne!(first_id, second_id);
        for captured in [captured_a, captured_b] {
            let parts = captured.lock().await;
            assert_eq!(
                parts.iter().find(|p| p.0 == "model").unwrap().3,
                "video-model"
            );
            assert_eq!(
                parts.iter().find(|p| p.0 == "extra_params").unwrap().3,
                "{\"task\":\"fl2va\",\"duration\":4}"
            );
            let media: Vec<_> = parts.iter().filter(|p| p.0 == "input_references").collect();
            assert_eq!(media.len(), 2);
            assert_eq!(media[0].1.as_deref(), Some("first.png"));
            assert_eq!(media[1].1.as_deref(), Some("last.png"));
            assert_eq!(media[0].2.as_deref(), Some("image/png"));
            assert_eq!(media[0].3, "first");
            assert_eq!(media[1].3, "last");
        }

        for id in [first_id, second_id] {
            let metadata = json_body(
                router
                    .handle_video_request(job_request(Method::GET, id, false))
                    .await,
            )
            .await;
            assert_eq!(metadata["id"], id);
        }
        router.video_workers.clear();
        let metadata = json_body(
            router
                .handle_video_request(job_request(Method::GET, second_id, false))
                .await,
        )
        .await;
        assert_eq!(metadata["id"], second_id);
        assert!(router.video_workers.contains_key(second_id));

        let head = router
            .handle_video_request(job_request(Method::HEAD, second_id, true))
            .await;
        assert_eq!(head.status(), StatusCode::OK);
        assert_eq!(head.headers()["content-type"], "video/mp4");
        assert_eq!(head.headers()["content-length"], "4");
        assert!(to_bytes(head.into_body(), 4096).await.unwrap().is_empty());
        let mut download = job_request(Method::GET, second_id, true);
        download
            .headers_mut()
            .insert(RANGE, "bytes=0-1".parse().unwrap());
        let download = router.handle_video_request(download).await;
        assert_eq!(download.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(download.headers()["content-range"], "bytes 0-1/4");
        assert_eq!(to_bytes(download.into_body(), 4096).await.unwrap(), "vi");

        let deleted = json_body(
            router
                .handle_video_request(job_request(Method::DELETE, first_id, false))
                .await,
        )
        .await;
        assert_eq!(deleted["id"], first_id);
        assert_eq!(deleted["deleted"], true);
        assert!(!router.video_workers.contains_key(first_id));
        assert_eq!(
            router
                .handle_video_request(job_request(Method::GET, first_id, false))
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            router
                .handle_video_request(job_request(Method::GET, second_id, false))
                .await
                .status(),
            StatusCode::OK
        );
        server_a.abort();
        server_b.abort();
    }
}
