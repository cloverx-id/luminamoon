use std::fmt;
use std::fs;
use std::path::Path;

use reqwest::Client;
use reqwest::StatusCode;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::json;

use crate::error::{DataLoaderError, Result};
use crate::multimodal::{MultimodalValue, encode_base64, serialize_columns_to_parquet_bytes};

pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

#[derive(Clone)]
pub struct HfCredentials {
    token: Option<String>,
    username: Option<String>,
    org: Option<String>,
    endpoint: String,
}

impl fmt::Debug for HfCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HfCredentials")
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .field("username", &self.username)
            .field("org", &self.org)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

impl Default for HfCredentials {
    fn default() -> Self {
        Self::from_env()
    }
}

impl HfCredentials {
    pub fn new(
        token: Option<String>,
        username: Option<String>,
        org: Option<String>,
        endpoint: Option<String>,
    ) -> Self {
        let env_token = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
            .or_else(Self::read_cached_token_file);
        let env_user = std::env::var("HF_USERNAME").ok();
        let env_org = std::env::var("HF_ORG").ok();
        let env_ep = std::env::var("HF_ENDPOINT").ok();

        let clean_opt =
            |opt: Option<String>| opt.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());

        let resolved_ep = clean_opt(endpoint)
            .or_else(|| clean_opt(env_ep))
            .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_owned())
            .trim_end_matches('/')
            .to_owned();

        Self {
            token: clean_opt(token).or_else(|| clean_opt(env_token)),
            username: clean_opt(username).or_else(|| clean_opt(env_user)),
            org: clean_opt(org).or_else(|| clean_opt(env_org)),
            endpoint: resolved_ep,
        }
    }

    pub fn from_env() -> Self {
        Self::new(None, None, None, None)
    }

    fn read_cached_token_file() -> Option<String> {
        if let Ok(hf_home) = std::env::var("HF_HOME") {
            let p = Path::new(&hf_home).join("token");
            if let Ok(t) = fs::read_to_string(p) {
                return Some(t);
            }
        }
        let home = std::env::var("HOME")
            .ok()
            .or_else(|| std::env::var("USERPROFILE").ok())?;
        let p = Path::new(&home)
            .join(".cache")
            .join("huggingface")
            .join("token");
        fs::read_to_string(p).ok()
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    pub fn username(&self) -> Option<&str> {
        self.username.as_deref()
    }

    pub fn org(&self) -> Option<&str> {
        self.org.as_deref()
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn default_owner(&self) -> Option<&str> {
        self.org.as_deref().or(self.username.as_deref())
    }

    pub fn resolve_repo_id(&self, repo: &str) -> Result<(String, String)> {
        let trimmed = repo.trim().trim_matches('/');
        if trimmed.is_empty() {
            return Err(DataLoaderError::InvalidConfig(
                "Repository name cannot be empty".to_owned(),
            ));
        }

        if let Some((owner, name)) = trimmed.split_once('/') {
            if owner.is_empty() || name.is_empty() || name.contains('/') {
                return Err(DataLoaderError::InvalidConfig(format!(
                    "Invalid repository identifier: {repo}"
                )));
            }
            return Ok((owner.to_owned(), name.to_owned()));
        }

        let owner = self.default_owner().ok_or_else(|| {
            DataLoaderError::InvalidConfig(format!(
                "Repository '{repo}' has no namespace prefix and neither HF_ORG nor HF_USERNAME is set"
            ))
        })?;

        Ok((owner.to_owned(), trimmed.to_owned()))
    }

    pub fn validate_url_security(&self, url: &str) -> Result<()> {
        if self.token.is_none() {
            return Ok(());
        }
        if url.starts_with("https://") {
            return Ok(());
        }
        if url.starts_with("http://127.0.0.1")
            || url.starts_with("http://localhost")
            || url.starts_with("http://[::1]")
        {
            return Ok(());
        }
        Err(DataLoaderError::InvalidConfig(
            "Refusing to transmit HF_TOKEN over unencrypted non-loopback HTTP connection"
                .to_owned(),
        ))
    }
}

#[derive(Clone)]
pub struct HfHubClient {
    client: Client,
    creds: HfCredentials,
}

impl HfHubClient {
    pub fn new(creds: HfCredentials) -> Result<Self> {
        creds.validate_url_security(creds.endpoint())?;
        let client = Client::builder().build()?;
        Ok(Self { client, creds })
    }

    pub fn credentials(&self) -> &HfCredentials {
        &self.creds
    }

    fn require_token(&self) -> Result<&str> {
        self.creds.token().ok_or_else(|| {
            DataLoaderError::InvalidConfig(
                "HF_TOKEN is required for HuggingFace Hub write/upload operations".to_owned(),
            )
        })
    }

    pub async fn create_repo(
        &self,
        repo: &str,
        repo_type: &str,
        private: bool,
        exist_ok: bool,
    ) -> Result<String> {
        let token = self.require_token()?;
        let (owner, name) = self.creds.resolve_repo_id(repo)?;
        let url = format!("{}/api/repos/create", self.creds.endpoint());
        self.creds.validate_url_security(&url)?;

        let mut payload = json!({
            "name": name,
            "type": repo_type,
            "private": private,
        });

        if self.creds.username() != Some(owner.as_str()) {
            payload["organization"] = json!(owner);
        }

        let resp = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .json(&payload)
            .send()
            .await?;

        if resp.status() == StatusCode::CONFLICT && exist_ok {
            return Ok(format!("{owner}/{name}"));
        }

        resp.error_for_status().map_err(|e| {
            DataLoaderError::InvalidConfig(format!(
                "HF create_repo failed: {}",
                e.status().unwrap_or_default()
            ))
        })?;

        Ok(format!("{owner}/{name}"))
    }

    pub async fn upload_bytes(
        &self,
        repo: &str,
        path_in_repo: &str,
        data: &[u8],
        revision: &str,
        commit_message: &str,
    ) -> Result<String> {
        let token = self.require_token()?;
        let (owner, name) = self.creds.resolve_repo_id(repo)?;
        let clean_path = path_in_repo.trim_start_matches('/');
        if clean_path.is_empty() || clean_path.contains("..") {
            return Err(DataLoaderError::InvalidConfig(format!(
                "Invalid path_in_repo: {path_in_repo}"
            )));
        }

        let rev = if revision.trim().is_empty() {
            "main"
        } else {
            revision.trim()
        };

        let url = format!(
            "{}/api/datasets/{owner}/{name}/commit/{rev}",
            self.creds.endpoint()
        );
        self.creds.validate_url_security(&url)?;

        let header_line = json!({
            "key": "header",
            "value": {
                "summary": commit_message,
                "description": ""
            }
        });
        let file_line = json!({
            "key": "file",
            "value": {
                "content": encode_base64(data),
                "path": clean_path,
                "encoding": "base64"
            }
        });

        let ndjson_body = format!("{header_line}\n{file_line}\n");

        let resp = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .header(CONTENT_TYPE, "application/x-ndjson")
            .body(ndjson_body)
            .send()
            .await?;

        resp.error_for_status().map_err(|e| {
            DataLoaderError::InvalidConfig(format!(
                "HF commit upload failed: {}",
                e.status().unwrap_or_default()
            ))
        })?;

        Ok(format!("hf://datasets/{owner}/{name}@{rev}/{clean_path}"))
    }

    pub async fn upload_file(
        &self,
        repo: &str,
        local_path: impl AsRef<Path>,
        path_in_repo: &str,
        revision: &str,
        commit_message: &str,
    ) -> Result<String> {
        let bytes = fs::read(local_path)?;
        self.upload_bytes(repo, path_in_repo, &bytes, revision, commit_message)
            .await
    }

    pub async fn upload_multimodal_parquet(
        &self,
        repo: &str,
        path_in_repo: &str,
        columns: &[(String, Vec<MultimodalValue>)],
        revision: &str,
        commit_message: &str,
    ) -> Result<String> {
        let pq_bytes = serialize_columns_to_parquet_bytes(columns)?;
        self.upload_bytes(repo, path_in_repo, &pq_bytes, revision, commit_message)
            .await
    }

    pub async fn upload_multimodal_jsonl(
        &self,
        repo: &str,
        path_in_repo: &str,
        columns: &[(String, Vec<MultimodalValue>)],
        revision: &str,
        commit_message: &str,
    ) -> Result<String> {
        let jsonl_bytes = crate::multimodal::serialize_columns_to_jsonl_bytes(columns)?;
        self.upload_bytes(repo, path_in_repo, &jsonl_bytes, revision, commit_message)
            .await
    }

    pub async fn upload_multimodal_json(
        &self,
        repo: &str,
        path_in_repo: &str,
        columns: &[(String, Vec<MultimodalValue>)],
        revision: &str,
        commit_message: &str,
    ) -> Result<String> {
        let json_bytes = crate::multimodal::serialize_columns_to_json_bytes(columns)?;
        self.upload_bytes(repo, path_in_repo, &json_bytes, revision, commit_message)
            .await
    }

    pub async fn upload_table_auto_format(
        &self,
        repo: &str,
        path_in_repo: &str,
        columns: &[(String, Vec<MultimodalValue>)],
        revision: &str,
        commit_message: &str,
        hf_image_struct: bool,
    ) -> Result<String> {
        let lower = path_in_repo.to_ascii_lowercase();
        let payload = if lower.ends_with(".jsonl") || lower.ends_with(".ndjson") {
            crate::multimodal::serialize_columns_to_jsonl_bytes(columns)?
        } else if lower.ends_with(".json") {
            crate::multimodal::serialize_columns_to_json_bytes(columns)?
        } else {
            crate::multimodal::serialize_columns_to_parquet_bytes_with_options(
                columns,
                hf_image_struct,
            )?
        };
        self.upload_bytes(repo, path_in_repo, &payload, revision, commit_message)
            .await
    }
}
