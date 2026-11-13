//! Built-in 3D Identity Space.
//!
//! Positions come from the "space" Python worker (PCA over the 512-d
//! face embeddings); identity assignment and asset paths come from
//! storage. Depth maps for the relief view are generated on demand by
//! the same worker when a Depth Anything V2 model is installed.

use std::collections::HashMap;
use std::path::Path;

use dahlia_core::jobs::{JobHooks, ProgressInfo, WorkerConfig, WorkerSession};
use dahlia_core::storage::{Db, Identity};
use serde::Serialize;
use serde_json::json;

use crate::pipeline::EMBED_MODEL;

/// Faces sent to the space worker per add_items request (W23).
const SPACE_CHUNK: usize = 500;

#[derive(Serialize)]
pub struct SpacePoint {
    pub face_id: i64,
    pub identity_id: Option<i64>,
    pub pos: [f64; 3],
    pub confidence: f32,
    pub asset_path: String,
    pub crop_path: String,
}

#[derive(Serialize)]
pub struct SpaceData {
    pub points: Vec<SpacePoint>,
    pub identities: Vec<Identity>,
    pub method: String,
}

pub fn identity_space(
    db: &Db,
    cfg: &WorkerConfig,
    crops_dir: &Path,
    project_id: i64,
) -> Result<SpaceData, String> {
    identity_space_hooked(db, cfg, crops_dir, project_id, &JobHooks::none())
}

pub fn identity_space_hooked(
    db: &Db,
    cfg: &WorkerConfig,
    crops_dir: &Path,
    project_id: i64,
    hooks: &JobHooks,
) -> Result<SpaceData, String> {
    let identities = db.list_identities(project_id).map_err(|e| e.to_string())?;
    let items = db
        .faces_with_embeddings(project_id, EMBED_MODEL)
        .map_err(|e| e.to_string())?;
    if items.is_empty() {
        return Ok(SpaceData { points: Vec::new(), identities, method: "pca".into() });
    }

    // face -> identity / asset path, via the identity detail query.
    let mut face_identity: HashMap<i64, i64> = HashMap::new();
    let mut face_asset: HashMap<i64, String> = HashMap::new();
    for ident in &identities {
        if let Ok(pairs) = db.identity_faces_detailed(ident.id) {
            for (face, asset) in pairs {
                face_identity.insert(face.id, ident.id);
                face_asset.insert(face.id, asset.path);
            }
        }
    }
    let assets = db.list_assets(project_id).map_err(|e| e.to_string())?;
    let asset_path: HashMap<i64, String> =
        assets.iter().map(|a| (a.id, a.path.clone())).collect();

    let mut session = WorkerSession::start(cfg, "space").map_err(|e| e.to_string())?;
    // Stream the faces in chunks instead of one giant request
    // (W23), reporting the send as the first half of the job; the
    // worker then reports the reduction as the second half.
    // Cancelling mid-send just drops the session: nothing has been
    // persisted yet, so there is no partial state to clean up.
    let n = items.len() as u64;
    let mut sent = 0u64;
    hooks.report(0, 2 * n, Some("データを送信中"));
    for chunk in items.chunks(SPACE_CHUNK) {
        let payload: Vec<_> = chunk
            .iter()
            .map(|f| json!({ "face_id": f.face_id, "vector": f.vector }))
            .collect();
        session
            .call_hooked(json!({ "op": "add_items", "items": payload }), hooks)
            .map_err(|e| e.to_string())?;
        sent += chunk.len() as u64;
        hooks.report(sent, 2 * n, Some("データを送信中"));
    }
    let compute_hooks = hooks.mapped(move |p: ProgressInfo| ProgressInfo {
        done: n + (p.done * n / p.total.max(1)).min(n),
        total: 2 * n,
        phase: Some("空間を計算中".to_string()),
    });
    let resp = session
        .call_hooked(json!({ "op": "reduce", "method": "pca" }), &compute_hooks)
        .map_err(|e| e.to_string())?;
    let method = resp
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("pca")
        .to_string();

    let mut pos_by_face: HashMap<i64, [f64; 3]> = HashMap::new();
    if let Some(arr) = resp.get("points").and_then(|v| v.as_array()) {
        for p in arr {
            let fid = p.get("face_id").and_then(|v| v.as_i64()).unwrap_or(0);
            let pos = p
                .get("pos")
                .and_then(|v| v.as_array())
                .map(|a| {
                    let g = |i: usize| a.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0);
                    [g(0), g(1), g(2)]
                })
                .unwrap_or([0.0; 3]);
            pos_by_face.insert(fid, pos);
        }
    }

    let points = items
        .iter()
        .filter_map(|f| {
            pos_by_face.get(&f.face_id).map(|pos| SpacePoint {
                face_id: f.face_id,
                identity_id: face_identity.get(&f.face_id).copied(),
                pos: *pos,
                confidence: f.confidence,
                asset_path: face_asset
                    .get(&f.face_id)
                    .cloned()
                    .or_else(|| asset_path.get(&f.asset_id).cloned())
                    .unwrap_or_default(),
                crop_path: crops_dir
                    .join(format!("{}.jpg", f.face_id))
                    .to_string_lossy()
                    .into_owned(),
            })
        })
        .collect();
    Ok(SpaceData { points, identities, method })
}

/// Generate (or reuse) the depth map PNG for one face crop.
/// Returns the PNG path; the frontend turns it into a relief mesh.
pub fn face_depth(cfg: &WorkerConfig, data_dir: &Path, face_id: i64) -> Result<String, String> {
    let out = data_dir.join("depths").join(format!("{face_id}.png"));
    if out.exists() {
        return Ok(out.to_string_lossy().into_owned());
    }
    let crop = data_dir.join("faces").join(format!("{face_id}.jpg"));
    if !crop.exists() {
        return Err("顔の切り抜き画像がありません（先に顔検出を実行してください）".into());
    }
    let mut session = WorkerSession::start(cfg, "space").map_err(|e| e.to_string())?;
    let resp = session
        .call(json!({
            "depth": {
                "image": crop.to_string_lossy(),
                "out": out.to_string_lossy(),
            }
        }))
        .map_err(|e| e.to_string())?;
    if resp.get("depth_ready").and_then(|v| v.as_bool()) == Some(true) {
        Ok(out.to_string_lossy().into_owned())
    } else {
        Err("深度モデル（Depth Anything V2）が見つかりません。models フォルダに depth-anything-v2-small.onnx を配置すると3Dレリーフが使えます".into())
    }
}
