//! Analysis pipeline: orchestrates the Python AI workers through
//! dahlia-core's job bridge and persists results via storage.
//!
//! Flow per project:
//!   detect_faces        -> faces (+ face crop thumbnails for the UI)
//!   generate_embeddings -> embeddings (embedding worker re-detects
//!                          internally for landmarks; results are
//!                          matched to stored faces by bbox center)
//!   cluster_identities  -> identities (DBSCAN) + quality scores

use std::path::Path;

use dahlia_core::identity::cosine;
use dahlia_core::jobs::{WorkerConfig, WorkerSession};
use dahlia_core::storage::{Db, NewIdentity};
use serde::Serialize;
use serde_json::{json, Value};

pub const EMBED_MODEL: &str = "arcface-mbf";
const CLUSTER_EPS: f64 = 0.68; // cosine distance; tuned on tests/samples

#[derive(Serialize)]
pub struct DetectReport {
    pub images_processed: usize,
    pub faces_found: usize,
    pub failed: usize,
}

#[derive(Serialize)]
pub struct EmbedReport {
    pub embedded: usize,
    pub failed: usize,
}

#[derive(Serialize)]
pub struct ClusterReport {
    pub identities: usize,
    pub clustered_faces: usize,
    pub single_faces: usize,
}

fn faces_from(resp: &Value) -> Vec<(dahlia_core::storage::NewFace, Option<Vec<f32>>)> {
    let mut out = Vec::new();
    if let Some(arr) = resp.get("faces").and_then(|v| v.as_array()) {
        for f in arr {
            let bbox = f
                .get("bbox")
                .and_then(|v| v.as_array())
                .map(|a| {
                    let g = |i: usize| a.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    [g(0), g(1), g(2), g(3)]
                })
                .unwrap_or([0.0; 4]);
            let confidence = f.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let embedding = f.get("embedding").and_then(|v| v.as_array()).map(|a| {
                a.iter().filter_map(|x| x.as_f64().map(|y| y as f32)).collect::<Vec<f32>>()
            });
            out.push((
                dahlia_core::storage::NewFace { bbox, confidence },
                embedding,
            ));
        }
    }
    out
}

fn save_crop(asset_path: &str, bbox: [f32; 4], dest: &Path) {
    let Ok(img) = image::open(asset_path) else { return };
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let x = bbox[0].clamp(0.0, iw - 1.0) as u32;
    let y = bbox[1].clamp(0.0, ih - 1.0) as u32;
    let w = bbox[2].clamp(1.0, iw - x as f32) as u32;
    let h = bbox[3].clamp(1.0, ih - y as f32) as u32;
    if w == 0 || h == 0 {
        return;
    }
    let crop = img.crop_imm(x, y, w, h);
    if let Some(parent) = dest.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = crop.save_with_format(dest, image::ImageFormat::Jpeg);
}

pub fn detect_faces(
    db: &Db,
    cfg: &WorkerConfig,
    crops_dir: &Path,
    project_id: i64,
) -> Result<DetectReport, String> {
    let assets = db.list_assets(project_id).map_err(|e| e.to_string())?;
    let mut session = WorkerSession::start(cfg, "face").map_err(|e| e.to_string())?;
    let (mut processed, mut found, mut failed) = (0usize, 0usize, 0usize);
    for asset in &assets {
        match session.call(json!({ "image": asset.path })) {
            Ok(resp) => {
                let detected = faces_from(&resp);
                let new_faces: Vec<_> = detected.into_iter().map(|(f, _)| f).collect();
                match db.replace_faces(asset.id, &new_faces) {
                    Ok(saved) => {
                        processed += 1;
                        found += saved.len();
                        for face in &saved {
                            save_crop(&asset.path, face.bbox, &crops_dir.join(format!("{}.jpg", face.id)));
                        }
                    }
                    Err(_) => failed += 1,
                }
            }
            Err(_) => failed += 1,
        }
    }
    Ok(DetectReport { images_processed: processed, faces_found: found, failed })
}

pub fn generate_embeddings(
    db: &Db,
    cfg: &WorkerConfig,
    project_id: i64,
) -> Result<EmbedReport, String> {
    let assets = db.list_assets(project_id).map_err(|e| e.to_string())?;
    let mut session = WorkerSession::start(cfg, "embedding").map_err(|e| e.to_string())?;
    let (mut embedded, mut failed) = (0usize, 0usize);
    for asset in &assets {
        let stored = db.list_faces(asset.id).map_err(|e| e.to_string())?;
        if stored.is_empty() {
            continue;
        }
        match session.call(json!({ "image": asset.path })) {
            Ok(resp) => {
                for (detected, embedding) in faces_from(&resp) {
                    let Some(vector) = embedding else { continue };
                    // Match to the stored face with the nearest bbox center.
                    let (dx, dy) = (
                        detected.bbox[0] + detected.bbox[2] / 2.0,
                        detected.bbox[1] + detected.bbox[3] / 2.0,
                    );
                    let nearest = stored.iter().min_by(|a, b| {
                        let da = ((a.bbox[0] + a.bbox[2] / 2.0) - dx).powi(2)
                            + ((a.bbox[1] + a.bbox[3] / 2.0) - dy).powi(2);
                        let db_ = ((b.bbox[0] + b.bbox[2] / 2.0) - dx).powi(2)
                            + ((b.bbox[1] + b.bbox[3] / 2.0) - dy).powi(2);
                        da.partial_cmp(&db_).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if let Some(face) = nearest {
                        match db.save_embedding(face.id, EMBED_MODEL, &vector) {
                            Ok(()) => embedded += 1,
                            Err(_) => failed += 1,
                        }
                    }
                }
            }
            Err(_) => failed += 1,
        }
    }
    Ok(EmbedReport { embedded, failed })
}

fn mean_cosine_to_centroid(vectors: &[&Vec<f32>]) -> f32 {
    if vectors.is_empty() {
        return 0.0;
    }
    let dim = vectors[0].len();
    let mut centroid = vec![0f32; dim];
    for v in vectors {
        for (i, x) in v.iter().enumerate() {
            centroid[i] += x;
        }
    }
    let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return 0.0;
    }
    for x in &mut centroid {
        *x /= norm;
    }
    vectors.iter().map(|v| cosine(v, &centroid)).sum::<f32>() / vectors.len() as f32
}

pub fn cluster_identities(
    db: &Db,
    cfg: &WorkerConfig,
    project_id: i64,
) -> Result<ClusterReport, String> {
    let items = db
        .faces_with_embeddings(project_id, EMBED_MODEL)
        .map_err(|e| e.to_string())?;
    if items.is_empty() {
        db.replace_identities(project_id, &[]).map_err(|e| e.to_string())?;
        return Ok(ClusterReport { identities: 0, clustered_faces: 0, single_faces: 0 });
    }
    let by_id: std::collections::HashMap<i64, &dahlia_core::storage::FaceEmbedding> =
        items.iter().map(|f| (f.face_id, f)).collect();

    let mut session = WorkerSession::start(cfg, "clustering").map_err(|e| e.to_string())?;
    let payload_items: Vec<Value> = items
        .iter()
        .map(|f| json!({ "face_id": f.face_id, "vector": f.vector }))
        .collect();
    let resp = session
        .call(json!({ "items": payload_items, "eps": CLUSTER_EPS, "min_samples": 2 }))
        .map_err(|e| e.to_string())?;

    let mut groups: Vec<Vec<i64>> = Vec::new();
    if let Some(clusters) = resp.get("clusters").and_then(|v| v.as_array()) {
        for c in clusters {
            let ids: Vec<i64> = c
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default();
            if !ids.is_empty() {
                groups.push(ids);
            }
        }
    }
    let noise: Vec<i64> = resp
        .get("noise")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default();

    // Larger clusters first; singleton (noise) faces become their own
    // identities so no detected face is ever lost.
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    let clustered_faces: usize = groups.iter().map(|g| g.len()).sum();
    let single_faces = noise.len();

    let mut new_identities: Vec<NewIdentity> = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let vectors: Vec<&Vec<f32>> = group
            .iter()
            .filter_map(|id| by_id.get(id).map(|f| &f.vector))
            .collect();
        new_identities.push(NewIdentity {
            name: format!("人物 {}", i + 1),
            quality_score: mean_cosine_to_centroid(&vectors),
            face_ids: group.clone(),
        });
    }
    for (j, face_id) in noise.iter().enumerate() {
        let quality = by_id.get(face_id).map(|f| f.confidence).unwrap_or(0.0);
        new_identities.push(NewIdentity {
            name: format!("人物 {}", groups.len() + j + 1),
            quality_score: quality,
            face_ids: vec![*face_id],
        });
    }

    let total = new_identities.len();
    db.replace_identities(project_id, &new_identities).map_err(|e| e.to_string())?;
    Ok(ClusterReport { identities: total, clustered_faces, single_faces })
}
