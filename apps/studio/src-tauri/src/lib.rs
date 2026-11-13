//! Dahlia Studio - Tauri application shell.
//!
//! All data and AI work goes through dahlia-core (storage / jobs);
//! the frontend only ever calls the commands defined here.

pub mod facemesh;
pub mod import;
pub mod pipeline;
pub mod space;

use std::path::PathBuf;
use std::sync::Arc;

use dahlia_core::jobs::WorkerConfig;
use dahlia_core::storage::{Asset, Db, Face, Identity, Project};
use serde::Serialize;
use tauri::{Manager, State};

pub struct AppState {
    db: Arc<Db>,
    data_dir: PathBuf,
    workers: WorkerConfig,
}

impl AppState {
    fn thumbs_dir(&self) -> PathBuf {
        self.data_dir.join("thumbnails")
    }
    fn face_crops_dir(&self) -> PathBuf {
        self.data_dir.join("faces")
    }
}

fn to_err<T>(r: dahlia_core::Result<T>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct DbStatus {
    ok: bool,
    db_path: String,
    tables: Vec<String>,
    /// Set when startup recovered the database from a backup.
    notice: Option<String>,
}

#[tauri::command]
fn db_status(state: State<'_, AppState>) -> Result<DbStatus, String> {
    let tables = to_err(state.db.table_names())?;
    Ok(DbStatus {
        ok: true,
        db_path: state.data_dir.join("dahlia.db").to_string_lossy().into_owned(),
        tables,
        notice: state.db.recovery_note().map(|s| s.to_string()),
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
    let thumbs = state.thumbs_dir();
    tauri::async_runtime::spawn_blocking(move || {
        import::import_folder(&db, &thumbs, project_id, &PathBuf::from(folder))
    })
    .await
    .map_err(|e| e.to_string())?
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
    tauri::async_runtime::spawn_blocking(move || {
        pipeline::detect_faces(&db, &workers, &crops, project_id)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn generate_embeddings(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<pipeline::EmbedReport, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    tauri::async_runtime::spawn_blocking(move || {
        pipeline::generate_embeddings(&db, &workers, project_id)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn cluster_identities(
    state: State<'_, AppState>,
    project_id: i64,
) -> Result<pipeline::ClusterReport, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    tauri::async_runtime::spawn_blocking(move || {
        pipeline::cluster_identities(&db, &workers, project_id)
    })
    .await
    .map_err(|e| e.to_string())?
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
            crop_path: crops.join(format!("{}.jpg", face.id)).to_string_lossy().into_owned(),
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
    tauri::async_runtime::spawn_blocking(move || {
        space::identity_space(&db, &workers, &crops, project_id)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn face_depth(state: State<'_, AppState>, face_id: i64) -> Result<String, String> {
    let workers = state.workers.clone();
    let data_dir = state.data_dir.clone();
    tauri::async_runtime::spawn_blocking(move || space::face_depth(&workers, &data_dir, face_id))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn identity_face_mesh(
    state: State<'_, AppState>,
    identity_id: i64,
) -> Result<facemesh::FaceMeshData, String> {
    let db = state.db.clone();
    let workers = state.workers.clone();
    let data_dir = state.data_dir.clone();
    tauri::async_runtime::spawn_blocking(move || {
        facemesh::identity_face_mesh(&db, &workers, &data_dir, identity_id)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn rename_identity(state: State<'_, AppState>, identity_id: i64, name: String) -> Result<(), String> {
    to_err(state.db.rename_identity(identity_id, &name))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&data_dir)?;
            let db = Db::open(&data_dir.join("dahlia.db"))?;
            let workers = WorkerConfig::locate();
            app.manage(AppState { db: Arc::new(db), data_dir, workers });
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
            identity_space,
            face_depth,
            identity_face_mesh
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
