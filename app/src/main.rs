use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    env,
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const WORKER_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct AppState {
    root: PathBuf,
    python: PathBuf,
    worker: Arc<Mutex<Option<Worker>>>,
    ready: Arc<AtomicBool>,
    sample: Option<Arc<Vec<u8>>>,
}

struct Worker {
    _child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

#[derive(Serialize)]
struct WorkerRequest {
    algorithm: String,
    multi_detection: bool,
    image_base64: String,
}

#[derive(Deserialize)]
struct WorkerReply {
    #[serde(default)]
    error: Option<String>,
    #[serde(flatten)]
    result: serde_json::Map<String, Value>,
}

impl Worker {
    async fn start(root: &PathBuf, python: &PathBuf) -> Result<Self, String> {
        let mut command = Command::new(python);
        command
            .arg("-u")
            .arg(root.join("scripts/app_worker.py"))
            .current_dir(root)
            .env("PYTHONIOENCODING", "utf-8")
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        #[cfg(windows)]
        command.creation_flags(0x0800_0000);
        let mut child = command
            .spawn()
            .map_err(|e| format!("Không chạy được Python: {e}"))?;
        let input = child.stdin.take().ok_or("Không mở được đầu vào mô hình")?;
        let output = child.stdout.take().ok_or("Không mở được đầu ra mô hình")?;
        let mut worker = Self {
            _child: child,
            input,
            output: BufReader::new(output),
        };
        let mut line = String::new();
        let count = timeout(WORKER_TIMEOUT, worker.output.read_line(&mut line))
            .await
            .map_err(|_| "Hết thời gian nạp mô hình".to_string())?
            .map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("Tiến trình mô hình đã dừng khi khởi động".into());
        }
        let ready: Value =
            serde_json::from_str(&line).map_err(|_| "Phản hồi khởi động không hợp lệ")?;
        if ready.get("ready") != Some(&Value::Bool(true)) {
            return Err(ready
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("Không nạp được mô hình")
                .into());
        }
        Ok(worker)
    }

    async fn recognize(&mut self, request: &WorkerRequest) -> Result<WorkerReply, String> {
        let mut message = serde_json::to_vec(request).map_err(|e| e.to_string())?;
        message.push(b'\n');
        self.input
            .write_all(&message)
            .await
            .map_err(|e| e.to_string())?;
        self.input.flush().await.map_err(|e| e.to_string())?;
        let mut line = String::new();
        if self
            .output
            .read_line(&mut line)
            .await
            .map_err(|e| e.to_string())?
            == 0
        {
            return Err("Tiến trình mô hình đã dừng".into());
        }
        serde_json::from_str(&line).map_err(|e| format!("Phản hồi mô hình không hợp lệ: {e}"))
    }
}

fn validate_algorithm(value: &str) -> Result<(), ApiError> {
    if matches!(value, "svm" | "pca_fld_hmm") {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Thuật toán không hợp lệ".into(),
        ))
    }
}

fn parse_multi(value: &str) -> Result<bool, ApiError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Chế độ nhiều biển số không hợp lệ".into(),
        )),
    }
}

async fn recognize(
    State(state): State<AppState>,
    mut form: Multipart,
) -> Result<Json<Value>, ApiError> {
    let mut algorithm = "svm".to_string();
    let mut multi_text = "false".to_string();
    let mut image = None;
    let mut invalid_field = false;
    let mut duplicate_image = false;
    while let Some(field) = form.next_field().await.map_err(|_| {
        ApiError(
            StatusCode::BAD_REQUEST,
            "Dữ liệu ảnh tải lên không hợp lệ".into(),
        )
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "algorithm" => {
                algorithm = field
                    .text()
                    .await
                    .map_err(|_| ApiError(StatusCode::BAD_REQUEST, "Thiếu thuật toán".into()))?;
            }
            "multi_detection" => {
                multi_text = field.text().await.map_err(|_| {
                    ApiError(StatusCode::BAD_REQUEST, "Thiếu chế độ nhận diện".into())
                })?;
            }
            "image" => {
                duplicate_image |= image.is_some();
                let bytes = field.bytes().await.map_err(|_| {
                    ApiError(StatusCode::BAD_REQUEST, "Không đọc được ảnh tải lên".into())
                })?;
                image = Some(bytes);
            }
            _ => {
                invalid_field = true;
                let _ = field.bytes().await;
            }
        }
    }
    // Consume the bounded multipart body before returning validation errors.
    // An early reply while a browser is still uploading can reset HTTP/1 sockets.
    validate_algorithm(&algorithm)?;
    let multi = parse_multi(&multi_text)?;
    if invalid_field || duplicate_image {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Mỗi yêu cầu chỉ nhận một ảnh và các lựa chọn hợp lệ".into(),
        ));
    }
    let bytes = image.ok_or(ApiError(
        StatusCode::BAD_REQUEST,
        "Vui lòng chọn ảnh".into(),
    ))?;
    if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Ảnh phải có dung lượng từ 1 byte đến 20 MB".into(),
        ));
    }
    let request = WorkerRequest {
        algorithm,
        multi_detection: multi,
        image_base64: STANDARD.encode(&bytes),
    };
    let mut guard = state.worker.try_lock().map_err(|_| {
        ApiError(
            StatusCode::CONFLICT,
            "Ứng dụng đang xử lý ảnh khác. Vui lòng thử lại.".into(),
        )
    })?;
    if guard.is_none() {
        *guard = Some(
            Worker::start(&state.root, &state.python)
                .await
                .map_err(|e| ApiError(StatusCode::SERVICE_UNAVAILABLE, e))?,
        );
        state.ready.store(true, Ordering::Relaxed);
    }
    let response = timeout(WORKER_TIMEOUT, guard.as_mut().unwrap().recognize(&request)).await;
    match response {
        Ok(Ok(reply)) => {
            if let Some(error) = reply.error {
                return Err(ApiError(StatusCode::BAD_REQUEST, error));
            }
            Ok(Json(Value::Object(reply.result)))
        }
        _ => {
            // Drop and kill the child before accepting another request: a late reply
            // from a timed-out image must never be reused for a different upload.
            *guard = None;
            state.ready.store(false, Ordering::Relaxed);
            Err(ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "Xử lý ảnh bị gián đoạn hoặc quá 60 giây. Vui lòng thử lại.".into(),
            ))
        }
    }
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({"ready": state.ready.load(Ordering::Relaxed),
        "busy": state.worker.try_lock().is_err(), "sample_available": state.sample.is_some(),
        "algorithms": ["svm", "pca_fld_hmm"]}))
}

async fn sample(State(state): State<AppState>) -> Result<Response, ApiError> {
    match state.sample {
        Some(bytes) => Ok((
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            bytes.as_ref().clone(),
        )
            .into_response()),
        None => Err(ApiError(
            StatusCode::NOT_FOUND,
            "Chưa có ảnh mẫu trong thư mục dự án".into(),
        )),
    }
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut root = env::var_os("PLATE_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .to_path_buf()
        });
    let mut port: u16 = 8787;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--project" => root = PathBuf::from(args.next().ok_or("Thiếu thư mục dự án")?),
            "--port" => port = args.next().ok_or("Thiếu cổng")?.parse()?,
            "--help" | "-h" => {
                println!("plate-detect-app [--project PROJECT_DIRECTORY] [--port 8787]");
                return Ok(());
            }
            _ => return Err(format!("Tham số không hợp lệ: {arg}").into()),
        }
    }
    // OpenCV's Haar loader cannot open Windows verbatim paths (\\?\D:\...).
    // Resolve relative paths without canonicalize() adding that prefix.
    root = std::path::absolute(root)?;
    if !root.is_dir() {
        return Err("Thư mục dự án không tồn tại".into());
    }
    let python = env::var_os("PLATE_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let local = root.join(if cfg!(windows) {
                ".venv/Scripts/python.exe"
            } else {
                ".venv/bin/python"
            });
            if local.exists() {
                local
            } else {
                PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
            }
        });
    println!("Đang nạp mô hình SVM và PCA + FLD + HMM...");
    let worker = Worker::start(&root, &python)
        .await
        .map_err(std::io::Error::other)?;
    let sample_bytes = std::fs::read(root.join("multi_dectec_sample.jpg"))
        .ok()
        .map(Arc::new);
    let state = AppState {
        root,
        python,
        worker: Arc::new(Mutex::new(Some(worker))),
        ready: Arc::new(AtomicBool::new(true)),
        sample: sample_bytes,
    };
    let app = Router::new()
        .route(
            "/",
            get(|| async { Html(include_str!("../web/index.html")) }),
        )
        .route("/api/health", get(health))
        .route("/api/sample", get(sample))
        .route("/api/recognize", post(recognize))
        .layer(DefaultBodyLimit::max(MAX_IMAGE_BYTES + 1024 * 1024))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("PlateDetect: http://127.0.0.1:{port}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_the_two_trained_algorithms() {
        assert!(validate_algorithm("svm").is_ok());
        assert!(validate_algorithm("pca_fld_hmm").is_ok());
        assert!(validate_algorithm("unknown").is_err());
        assert!(validate_algorithm("").is_err());
    }

    #[test]
    fn multi_detection_is_explicit_not_truthy_string_parsing() {
        assert!(parse_multi("true").unwrap());
        assert!(!parse_multi("false").unwrap());
        assert!(parse_multi("FALSE").is_err());
        assert!(parse_multi("0").is_err());
    }
}
