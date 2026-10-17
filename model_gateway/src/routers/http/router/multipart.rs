//! Multipart uploads shared by image editing and video generation.

use axum::{
    body::Body,
    extract::{FromRequest, Multipart, Request},
    response::{IntoResponse, Response},
    Form,
};
use reqwest::multipart::Part;

use crate::routers::error;

pub(super) struct MultipartUpload {
    pub(super) model: String,
    pub(super) prompt: Option<String>,
    pub(super) parts: Vec<(String, Part)>,
}

impl MultipartUpload {
    pub(super) async fn parse_urlencoded(req: Request<Body>) -> Result<Self, Response> {
        let Form(fields) = Form::<Vec<(String, String)>>::from_request(req, &())
            .await
            .map_err(IntoResponse::into_response)?;
        let mut model = String::new();
        let mut prompt = None;
        let mut parts = Vec::new();
        for (name, value) in fields {
            if name == "model" {
                model = value;
                continue;
            }
            if name == "prompt" {
                prompt = Some(value.clone());
            }
            parts.push((name, Part::text(value)));
        }
        if model.trim().is_empty() {
            return Err(error::bad_request(
                "missing_model",
                "Missing required 'model' field",
            ));
        }
        Ok(Self {
            model: model.trim().to_owned(),
            prompt,
            parts,
        })
    }

    pub(super) async fn parse(req: Request<Body>) -> Result<Self, Response> {
        let mut multipart = Multipart::from_request(req, &())
            .await
            .map_err(IntoResponse::into_response)?;
        let mut model = String::new();
        let mut prompt = None;
        let mut parts = Vec::new();
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(IntoResponse::into_response)?
        {
            let name = field.name().unwrap_or_default().to_owned();
            let filename = field.file_name().map(str::to_owned);
            let content_type = field.content_type().map(str::to_owned);
            if name == "model" && filename.is_none() {
                model = field.text().await.map_err(IntoResponse::into_response)?;
                continue;
            }
            let bytes = field.bytes().await.map_err(IntoResponse::into_response)?;
            if name == "prompt" && filename.is_none() {
                prompt = std::str::from_utf8(&bytes).ok().map(str::to_owned);
            }
            let mut part = Part::stream(reqwest::Body::from(bytes));
            if let Some(filename) = filename {
                part = part.file_name(filename);
            }
            if let Some(content_type) = content_type {
                part = part
                    .mime_str(&content_type)
                    .map_err(|e| error::bad_request("invalid_multipart", e.to_string()))?;
            }
            parts.push((name, part));
        }
        if model.trim().is_empty() {
            return Err(error::bad_request(
                "missing_model",
                "Missing required 'model' field",
            ));
        }
        Ok(Self {
            model: model.trim().to_owned(),
            prompt,
            parts,
        })
    }
}
