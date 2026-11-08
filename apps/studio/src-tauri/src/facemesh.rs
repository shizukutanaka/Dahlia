//! 3D face preview for an identity.
//!
//! Picks the identity's sharpest face (highest detection confidence),
//! runs the "facemesh" worker (MediaPipe FaceLandmarker, 478
//! landmarks) over its crop, and caches the result as JSON under
//! <data_dir>/facemesh/ so each face is only processed once.

use std::path::Path;

use dahlia_core::jobs::{WorkerConfig, WorkerSession};
use dahlia_core::storage::Db;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize)]
pub struct FaceMeshData {
    pub face_id: i64,
    pub crop_path: String,
    pub points: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub blendshapes: Vec<(String, f32)>,
}

pub fn identity_face_mesh(
    db: &Db,
    cfg: &WorkerConfig,
    data_dir: &Path,
    identity_id: i64,
) -> Result<FaceMeshData, String> {
    let pairs = db
        .identity_faces_detailed(identity_id)
        .map_err(|e| e.to_string())?;
    let (face, _asset) = pairs
        .into_iter()
        .max_by(|a, b| a.0.confidence.partial_cmp(&b.0.confidence).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or("この人物には顔が登録されていません")?;

    let crop_path = data_dir.join("faces").join(format!("{}.jpg", face.id));
    let cache_path = data_dir.join("facemesh").join(format!("{}.json", face.id));
    if let Ok(text) = std::fs::read_to_string(&cache_path) {
        if let Ok(data) = serde_json::from_str::<FaceMeshData>(&text) {
            return Ok(data);
        }
    }
    if !crop_path.exists() {
        return Err("顔の切り抜き画像がありません（先に顔検出を実行してください）".into());
    }

    let mut session = WorkerSession::start(cfg, "facemesh").map_err(|e| e.to_string())?;
    let resp = session
        .call(json!({ "image": crop_path.to_string_lossy() }))
        .map_err(|e| e.to_string())?;
    if resp.get("found").and_then(|v| v.as_bool()) != Some(true) {
        return Err(
            "この顔では3Dランドマークを検出できませんでした（横顔や小さい顔では検出できないことがあります）"
                .into(),
        );
    }

    let points: Vec<[f32; 3]> = resp
        .get("points")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    let a = p.as_array()?;
                    Some([
                        a.first()?.as_f64()? as f32,
                        a.get(1)?.as_f64()? as f32,
                        a.get(2)?.as_f64()? as f32,
                    ])
                })
                .collect()
        })
        .unwrap_or_default();
    let uvs: Vec<[f32; 2]> = resp
        .get("uvs")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    let a = p.as_array()?;
                    Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32])
                })
                .collect()
        })
        .unwrap_or_default();
    let triangles: Vec<[u32; 3]> = resp
        .get("triangles")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let a = t.as_array()?;
                    Some([
                        a.first()?.as_u64()? as u32,
                        a.get(1)?.as_u64()? as u32,
                        a.get(2)?.as_u64()? as u32,
                    ])
                })
                .collect()
        })
        .unwrap_or_default();
    let mut blendshapes: Vec<(String, f32)> = resp
        .get("blendshapes")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0) as f32))
                .collect()
        })
        .unwrap_or_default();
    blendshapes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if points.len() < 100 {
        return Err("ランドマークの取得結果が不完全でした".into());
    }
    let data = FaceMeshData {
        face_id: face.id,
        crop_path: crop_path.to_string_lossy().into_owned(),
        points,
        uvs,
        triangles,
        blendshapes,
    };
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string(&data) {
        let _ = std::fs::write(&cache_path, text);
    }
    Ok(data)
}
