//! Axum extractors for multipart/form-data inference endpoints.
//!
//! JSON endpoints get [`crate::validated::ValidatedJson`]; the
//! `/v1/audio/transcriptions` endpoint uses multipart/form-data and gets
//! [`AudioTranscriptionMultipart`], which parses the form into a typed
//! `(TranscriptionRequest, AudioFile)` pair before the handler runs.

#[cfg(feature = "axum")]
use axum::{
    extract::{multipart::MultipartError, FromRequest, Multipart, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[cfg(feature = "axum")]
use crate::transcription::{AudioFile, MultipartTextField, TranscriptionRequest};

/// Extractor for `/v1/audio/transcriptions` requests.
///
/// Parses `multipart/form-data` into a [`TranscriptionRequest`] (text fields)
/// plus an [`AudioFile`] (the `file` part). Returns `400 Bad Request` on
/// malformed parts, missing/empty `file`, missing/blank `model`, or
/// out-of-range `temperature`.
#[cfg(feature = "axum")]
pub struct AudioTranscriptionMultipart {
    pub request: TranscriptionRequest,
    pub audio: AudioFile,
}

#[cfg(feature = "axum")]
impl<S: Send + Sync> FromRequest<S> for AudioTranscriptionMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(IntoResponse::into_response)?;

        let mut file_bytes: Option<bytes::Bytes> = None;
        let mut file_name: Option<String> = None;
        let mut file_content_type: Option<String> = None;
        let mut request = TranscriptionRequest::default();
        let mut timestamp_granularities: Vec<String> = Vec::new();
        let mut file_seen = false;

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(format!("Failed to read multipart field: {e}")));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            if name == "file" {
                if file_seen {
                    return Err(bad_request(
                        "Multiple 'file' parts are not supported".to_string(),
                    ));
                }
                file_seen = true;
                file_name = field.file_name().map(str::to_string);
                file_content_type = field.content_type().map(str::to_string);
                match field.bytes().await {
                    Ok(b) => file_bytes = Some(b),
                    Err(e) => {
                        return Err(bad_request(format!("Failed to read audio file bytes: {e}")));
                    }
                }
                continue;
            }

            if field.file_name().is_some() {
                let _ = field.bytes().await;
                continue;
            }

            let text = field.text().await.map_err(|e| bad_text_field(&name, e))?;
            request.multipart_fields.push(MultipartTextField {
                name: name.clone(),
                value: text.clone(),
            });
            match name.as_str() {
                "model" => request.model = text,
                "language" => request.language = Some(text),
                "prompt" => request.prompt = Some(text),
                "response_format" => request.response_format = Some(text),
                "temperature" => match text.trim().parse::<f32>() {
                    Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {
                        request.temperature = Some(v);
                    }
                    Ok(v) => {
                        return Err(bad_request(format!(
                                "Invalid 'temperature' value: {v} (must be a finite number in [0.0, 1.0])"
                            )));
                    }
                    Err(e) => {
                        return Err(bad_request(format!("Invalid 'temperature' value: {e}")));
                    }
                },
                "timestamp_granularities" | "timestamp_granularities[]" => {
                    timestamp_granularities.push(text);
                }
                "stream" => match text.as_str() {
                    "true" | "True" | "TRUE" | "1" => request.stream = Some(true),
                    "false" | "False" | "FALSE" | "0" => request.stream = Some(false),
                    other => {
                        return Err(bad_request(format!(
                            "Invalid 'stream' value: '{other}' (expected true/false/1/0)"
                        )));
                    }
                },
                _ => {}
            }
        }

        if request.model.trim().is_empty() {
            return Err(bad_request("Missing required 'model' field".to_string()));
        }
        request.model = request.model.trim().to_string();

        let bytes = match file_bytes {
            Some(b) if !b.is_empty() => b,
            Some(_) => {
                return Err(bad_request("Uploaded 'file' part is empty".to_string()));
            }
            None => {
                return Err(bad_request("Missing required 'file' part".to_string()));
            }
        };

        if !timestamp_granularities.is_empty() {
            request.timestamp_granularities = Some(timestamp_granularities);
        }

        let audio = AudioFile {
            bytes,
            file_name: file_name.unwrap_or_else(|| "audio".to_string()),
            content_type: file_content_type,
        };

        Ok(AudioTranscriptionMultipart { request, audio })
    }
}

#[cfg(feature = "axum")]
fn bad_request(message: String) -> Response {
    (StatusCode::BAD_REQUEST, message).into_response()
}

#[cfg(feature = "axum")]
fn bad_text_field(field: &str, e: MultipartError) -> Response {
    bad_request(format!("Failed to read '{field}' field: {e}"))
}

#[cfg(all(test, feature = "axum"))]
mod tests {
    use axum::{body::Body, extract::FromRequest, http::Request};

    use super::*;

    fn request(parts: &str) -> Request<Body> {
        Request::builder()
            .header(
                "content-type",
                "multipart/form-data; boundary=test-boundary",
            )
            .body(Body::from(parts.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn preserves_unknown_text_and_ignores_unknown_file_parts() {
        let request = request(concat!(
            "--test-boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nasr\r\n",
            "--test-boundary\r\nContent-Disposition: form-data; name=\"repetition_penalty\"\r\n\r\n1.1\r\n",
            "--test-boundary\r\nContent-Disposition: form-data; name=\"sidecar\"; filename=\"note.txt\"\r\n\r\nignore me\r\n",
            "--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n",
            "--test-boundary--\r\n"
        ));

        let parsed = AudioTranscriptionMultipart::from_request(request, &())
            .await
            .unwrap();
        assert_eq!(parsed.request.multipart_fields.len(), 2);
        assert_eq!(
            parsed.request.multipart_fields[1].name,
            "repetition_penalty"
        );
        assert_eq!(parsed.audio.file_name, "audio.wav");
        assert_eq!(parsed.audio.content_type.as_deref(), Some("audio/wav"));
    }

    #[tokio::test]
    async fn rejects_multiple_file_parts() {
        let request = request(concat!(
            "--test-boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nasr\r\n",
            "--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"one.wav\"\r\n\r\none\r\n",
            "--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"two.wav\"\r\n\r\ntwo\r\n",
            "--test-boundary--\r\n"
        ));

        let response = match AudioTranscriptionMultipart::from_request(request, &()).await {
            Ok(_) => panic!("multiple file parts must be rejected"),
            Err(response) => response,
        };
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
