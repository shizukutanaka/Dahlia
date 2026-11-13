//! Dahlia Studio - Tauri application shell.
//!
//! All data and AI work goes through dahlia-core (storage / jobs);
//! the frontend only ever calls the commands defined here.

pub mod errors;
pub mod facemesh;
pub mod heif;
pub mod import;
pub mod orient;
pub mod pipeline;
pub mod space;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use dahlia_core::jobs::{JobHooks, WorkerConfig};
use dahlia_core::storage::{Asset, Db, Face, Identity, Project};
use serde::Serialize;
use tauri::{Manager, State};

pub struct AppState {
    db: Arc<Db>,
    data_dir: PathBuf,
    workers: WorkerConfig,
    job: Arc<JobControl>,
}

/// The long-running job the UI is currently driving (the UI runs
/// one at a time): a cancel flag its hooks watch, and the latest
/// progress the frontend polls and displays (W9/W10).
pub struct JobControl {
    cancel: Mutex<Option<Arc<AtomicBool>>>,
    progress: Mutex<Option<JobProgress>>,
}

#[derive(Serialize, Clone)]
struct JobProgress {
    job: String,
    phase: Option<String>,
    done: u64,
    total: u64,
}

impl AppState {
    fn thumbs_dir(&self) -> PathBuf {
        self.data_dir.join("thumbnails")
    }
    fn face_crops_dir(&self) -> PathBuf {
        self.data_dir.join("faces")
    }
    /// JPEG working copies of HEIC photos (W87), keyed by asset id.
    fn proxies_dir(&self) -> PathBuf {
        self.data_dir.join("proxies")
    }

    /// Start tracking a job: fresh cancel flag, and hooks whose
    /// progress reports land in the shared slot the UI polls.
    fn begin_job(&self, name: &str) -> JobHooks {
        let flag = Arc::new(AtomicBool::new(false));
        *self.job.cancel.lock().unwrap() = Some(flag.clone());
        *self.job.progress.lock().unwrap() = Some(JobProgress {
            job: name.to_string(),
            phase: None,
            done: 0,
            total: 0,
        });
        let job = self.job.clone();
        let name = name.to_string();
        JobHooks {
            cancel: Some(flag),
            progress: Some(Arc::new(move |p: dahlia_core::jobs::ProgressInfo| {
                if let Ok(mut slot) = job.progress.lock() {
                    *slot = Some(JobProgress {
                        job: name.clone(),
                        phase: p.phase.clone(),
                        done: p.done,
                        total: p.total,
                    });
                }
            })),
        }
    }

    fn end_job(&self) {
        *self.job.cancel.lock().unwrap() = None;
        *self.job.progress.lock().unwrap() = None;
    }
}

/// Ask the running job (if any) to stop at the next safe point.
/// Work already completed stays saved.
#[tauri::command]
fn cancel_job(state: State<'_, AppState>) {
    if let Some(flag) = state.job.cancel.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
}

#[tauri::command]
fn job_progress(state: State<'_, AppState>) -> Option<JobProgress> {
    state.job.progress.lock().unwrap().clone()
}

fn to_err<T>(r: dahlia_core::Result<T>) -> Result<T, String> {
    r.map_err(|e| {
        let raw = e.to_string();
        dahlia_core::logging::error(&raw);
        errors::friendly(&raw)
    })
}

/// Same treatment for the String errors of the pipeline modules:
/// log the raw detail, show the user the Japanese version.
fn friendly<T>(r: Result<T, String>) -> Result<T, String> {
    r.map_err(|e| {
        dahlia_core::logging::error(&e);
        errors::friendly(&e)
    })
}

/// Open an image with its EXIF orientation applied (W86), so Rust-side
/// pixels agree with what the Python workers see (they apply the same
/// correction through PIL). Used for thumbnails and face crops.
pub(crate) fn open_oriented_image(path: &std::path::Path) -> Result<image::DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
    Ok(apply_orientation(img, orient::transform_for_jpeg(&bytes)))
}

fn apply_orientation(
    img: image::DynamicImage,
    transform: orient::Transform,
) -> image::DynamicImage {
    use orient::Transform;
    match transform {
        Transform::None => img,
        Transform::FlipHorizontal => img.fliph(),
        Transform::Rotate180 => img.rotate180(),
        Transform::FlipVertical => img.flipv(),
        Transform::Transpose => img.rotate90().fliph(),
        Transform::Rotate90Cw => img.rotate90(),
        Transform::Transverse => img.rotate90().flipv(),
        Transform::Rotate270Cw => img.rotate270(),
    }
}

#[derive(Serialize)]
struct DbStatus {
    ok: bool,
    db_path: String,
    tables: Vec<String>,
    /// Set when startup recovered the database from a backup.
    notice: Option<String>,
    /// App version (single source: this crate's Cargo.toml, kept in
    /// sync with tauri.conf.json / package.json).
    app_version: String,
    /// Where the diagnostic log lives (W12/W13).
    log_path: Option<String>,
}

#[tauri::command]
fn db_status(state: State<'_, AppState>) -> Result<DbStatus, String> {
    let tables = to_err(state.db.table_names())?;
    Ok(DbStatus {
        ok: true,
        db_path: state
            .data_dir
            .join("dahlia.db")
            .to_string_lossy()
            .into_owned(),
        tables,
        notice: state.db.recovery_note().map(|s| s.to_string()),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        log_path: dahlia_core::logging::log_path().map(|p| p.to_string_lossy().into_owned()),
    })
}

#[tauri::command]
fn create_project(state: State<'_, AppState>, name: String) -> Result<Project, String> {
    to_err(state.db.create_project(&name))
}

#[tauri::command]
fn list_projects(state: State<'_, AppState>) -> Result<Vec<Project>, String> {
    to_err(state.db.list_projects())
}

#[tauri::command]
async fn import_folder(
    state: State<'_, AppState>,
    project_id: i64,
    folder: String,
) -> Result<import::ImportReport, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let thumbs = state.thumbs_dir();
    let proxies = state.proxies_dir();
    let hooks = state.begin_job("写真の読み込み");
    let result = tauri::async_runtime::spawn_blocking(move || {
        import::import_folder_hooked(
            &db,
            &workers,
            &thumbs,
            &proxies,
            project_id,
            &PathBuf::from(folder),
            &hooks,
        )
    })
    .await
    .map_err(|e| e.to_string())?;
    state.end_job();
    friendly(result)
}

#[tauri::command]
fn list_assets(state: State<'_, AppState>, project_id: i64) -> Result<Vec<Asset>, String> {
    to_err(state.db.list_assets(project_id))
}

#[tauri::command]
fn list_faces(state: State<'_, AppState>, asset_id: i64) -> Result<Vec<Face>, String> {
    to_err(state.db.list_faces(asset_id))
}

#[tauri::command]
async fn detect_faces(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<pipeline::DetectReport, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let crops = state.face_crops_dir();
    let proxies = state.proxies_dir();
    let hooks = state.begin_job("顔の検出");
    let result = tauri::async_runtime::spawn_blocking(move || {
        pipeline::detect_faces_hooked(&db, &workers, &crops, &proxies, project_id, &hooks)
    })
    .await
    .map_err(|e| e.to_string())?;
    state.end_job();
    friendly(result)
}

#[tauri::command]
async fn generate_embeddings(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<pipeline::EmbedReport, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let hooks = state.begin_job("特徴量の取得");
    let result = tauri::async_runtime::spawn_blocking(move || {
        pipeline::generate_embeddings_hooked(&db, &workers, project_id, &hooks)
    })
    .await
    .map_err(|e| e.to_string())?;
    state.end_job();
    friendly(result)
}

#[tauri::command]
async fn cluster_identities(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<pipeline::ClusterReport, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let hooks = state.begin_job("人物の分類");
    let result = tauri::async_runtime::spawn_blocking(move || {
        pipeline::cluster_identities_hooked(&db, &workers, project_id, &hooks)
    })
    .await
    .map_err(|e| e.to_string())?;
    state.end_job();
    friendly(result)
}

#[tauri::command]
fn list_identities(state: State<'_, AppState>, project_id: i64) -> Result<Vec<Identity>, String> {
    to_err(state.db.list_identities(project_id))
}

#[derive(Serialize)]
struct FaceWithAsset {
    face: Face,
    asset_path: String,
    crop_path: String,
}

#[derive(Serialize)]
struct IdentityDetail {
    identity: Identity,
    faces: Vec<FaceWithAsset>,
}

#[tauri::command]
fn identity_detail(state: State<'_, AppState>, identity_id: i64) -> Result<IdentityDetail, String> {
    let identity = to_err(state.db.get_identity(identity_id))?;
    let pairs = to_err(state.db.identity_faces_detailed(identity_id))?;
    let crops = state.face_crops_dir();
    let faces = pairs
        .into_iter()
        .map(|(face, asset)| FaceWithAsset {
            crop_path: crops
                .join(format!("{}.jpg", face.id))
                .to_string_lossy()
                .into_owned(),
            asset_path: asset.path.clone(),
            face,
        })
        .collect();
    Ok(IdentityDetail { identity, faces })
}

#[tauri::command]
async fn identity_space(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<space::SpaceData, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let crops = state.face_crops_dir();
    let hooks = state.begin_job("3D空間の生成");
    let result = tauri::async_runtime::spawn_blocking(move || {
        space::identity_space_hooked(&db, &workers, &crops, project_id, &hooks)
    })
    .await
    .map_err(|e| e.to_string())?;
    state.end_job();
    friendly(result)
}

#[tauri::command]
async fn face_depth(state: State<'_, AppState>, face_id: i64) -> Result<String, String> {
    let workers = state.workers.clone();
    let data_dir = state.data_dir.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        space::face_depth(&workers, &data_dir, face_id)
    })
    .await
    .map_err(|e| e.to_string())?;
    friendly(result)
}

#[tauri::command]
async fn identity_face_mesh(
    state: State<'_, AppState>,
    identity_id: i64,
) -> Result<facemesh::FaceMeshData, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let data_dir = state.data_dir.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        facemesh::identity_face_mesh(&db, &workers, &data_dir, identity_id)
    })
    .await
    .map_err(|e| e.to_string())?;
    friendly(result)
}

#[tauri::command]
fn rename_identity(
    state: State<'_, AppState>,
    identity_id: i64,
    name: String,
) -> Result<(), String> {
    to_err(state.db.rename_identity(identity_id, &name))
}

#[tauri::command]
async fn redetect_faces(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<pipeline::DetectReport, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let crops = state.face_crops_dir();
    let proxies = state.proxies_dir();
    let hooks = state.begin_job("顔の検出（全件）");
    let result = tauri::async_runtime::spawn_blocking(move || {
        pipeline::redetect_faces_hooked(&db, &workers, &crops, &proxies, project_id, &hooks)
    })
    .await
    .map_err(|e| e.to_string())?;
    state.end_job();
    friendly(result)
}

#[tauri::command]
fn list_failures(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<Vec<dahlia_core::storage::ProcessingFailure>, String> {
    to_err(state.db.list_failures(project_id))
}

#[tauri::command]
fn clear_failures(state: State<'_, AppState>, project_id: i64) -> Result<(), String> {
    to_err(state.db.clear_failures(project_id))
}

/// Delete a project and its registrations. Generated files
/// (thumbnails, face crops) are removed; the user's original
/// photo files are never touched.
#[tauri::command]
fn delete_project(state: State<'_, AppState>, project_id: i64) -> Result<(), String> {
    let artifacts = to_err(state.db.project_artifacts(project_id))?;
    to_err(state.db.delete_project(project_id))?;
    for thumb in &artifacts.thumbnails {
        let _ = std::fs::remove_file(thumb);
    }
    let crops = state.face_crops_dir();
    for face_id in &artifacts.face_ids {
        let _ = std::fs::remove_file(crops.join(format!("{face_id}.jpg")));
    }
    dahlia_core::logging::info(&format!("プロジェクト {project_id} を削除しました"));
    Ok(())
}

/// Remove one photo's registration (faces/embeddings/links go with
/// it). The original file stays on disk.
#[tauri::command]
fn remove_asset(state: State<'_, AppState>, asset_id: i64) -> Result<(), String> {
    let removed = to_err(state.db.remove_asset(asset_id))?;
    if let Some(thumb) = &removed.thumbnail {
        let _ = std::fs::remove_file(thumb);
    }
    let crops = state.face_crops_dir();
    for face_id in &removed.face_ids {
        let _ = std::fs::remove_file(crops.join(format!("{face_id}.jpg")));
    }
    Ok(())
}

#[tauri::command]
fn merge_identities(
    state: State<'_, AppState>,
    target_id: i64,
    source_id: i64,
) -> Result<(), String> {
    to_err(state.db.merge_identities(target_id, source_id))
}

#[tauri::command]
fn move_face_to_identity(
    state: State<'_, AppState>,
    face_id: i64,
    target_identity_id: i64,
) -> Result<(), String> {
    to_err(state.db.move_face_to_identity(face_id, target_identity_id))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&data_dir)?;
            // Diagnostic log (local file only, never sent anywhere).
            dahlia_core::logging::init(&data_dir.join("logs").join("dahlia.log"));
            dahlia_core::logging::info("Dahlia を起動しました");
            let db = Db::open(&data_dir.join("dahlia.db"))?;
            if let Some(note) = db.recovery_note() {
                dahlia_core::logging::warn(note);
            }
            let workers = WorkerConfig::locate();
            app.manage(AppState {
                db: Arc::new(db),
                data_dir,
                workers,
                job: Arc::new(JobControl {
                    cancel: Mutex::new(None),
                    progress: Mutex::new(None),
                }),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            db_status,
            create_project,
            list_projects,
            import_folder,
            list_assets,
            list_faces,
            detect_faces,
            generate_embeddings,
            cluster_identities,
            list_identities,
            identity_detail,
            rename_identity,
            redetect_faces,
            list_failures,
            clear_failures,
            delete_project,
            remove_asset,
            merge_identities,
            move_face_to_identity,
            identity_space,
            face_depth,
            identity_face_mesh,
            cancel_job,
            job_progress
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
