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

use dahlia_core::identity::{PreviousIdentity, cosine, match_previous};
use dahlia_core::jobs::{WorkerConfig, WorkerSession};
use dahlia_core::storage::{Db, ReconciledIdentity};
use serde::Serialize;
use serde_json::{Value, json};

pub const EMBED_MODEL: &str = "arcface-mbf";
const CLUSTER_EPS: f64 = 0.68; // cosine distance; tuned on tests/samples

#[derive(Serialize)]
pub struct DetectReport {
    pub images_processed: usize,
    pub faces_found: usize,
    pub failed: usize,
    /// Assets left alone because detection already ran for them.
    pub skipped: usize,
}

#[derive(Serialize)]
pub struct EmbedReport {
    pub embedded: usize,
    pub failed: usize,
    /// Faces that already had an embedding and were not redone.
    pub skipped: usize,
}

#[derive(Serialize)]
pub struct ClusterReport {
    pub identities: usize,
    pub clustered_faces: usize,
    pub single_faces: usize,
    /// How many identities kept their previous (user-given) name.
    pub names_carried_over: usize,
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
                a.iter()
                    .filter_map(|x| x.as_f64().map(|y| y as f32))
                    .collect::<Vec<f32>>()
            });
            out.push((
                dahlia_core::storage::NewFace { bbox, confidence },
                embedding,
            ));
        }
    }
    out
}

fn save_crop(asset_path: &str, bbox: [f32; 4], dest: &Path) -> Result<(), String> {
    // BBoxes come from the workers, which see the EXIF-corrected
    // image; the crop must be cut from the same corrected pixels.
    // Failures are returned, not swallowed (W99): the caller records
    // them so "no crop" never appears out of nowhere downstream.
    let img = crate::open_oriented_image(Path::new(asset_path))?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let x = bbox[0].clamp(0.0, iw - 1.0) as u32;
    let y = bbox[1].clamp(0.0, ih - 1.0) as u32;
    let w = bbox[2].clamp(1.0, iw - x as f32) as u32;
    let h = bbox[3].clamp(1.0, ih - y as f32) as u32;
    if w == 0 || h == 0 {
        return Err("顔の切り抜き範囲を計算できませんでした".into());
    }
    let crop = img.crop_imm(x, y, w, h);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    crop.save_with_format(dest, image::ImageFormat::Jpeg)
        .map_err(|e| e.to_string())
}

/// Record a per-asset failure (best effort: recording must not mask
/// the original outcome) with a user-facing message (W11).
fn record(db: &Db, project_id: i64, asset_id: i64, stage: &str, target: &str, raw: &str) {
    let _ = db.record_failure(
        project_id,
        Some(asset_id),
        stage,
        target,
        &crate::errors::friendly(raw),
    );
}

pub fn detect_faces(
    db: &Db,
    cfg: &WorkerConfig,
    crops_dir: &Path,
    project_id: i64,
) -> Result<DetectReport, String> {
    detect_impl(db, cfg, crops_dir, project_id, false)
}

/// Detection that deliberately redoes every photo, including the
/// ones already processed (the normal run skips those).
pub fn redetect_faces(
    db: &Db,
    cfg: &WorkerConfig,
    crops_dir: &Path,
    project_id: i64,
) -> Result<DetectReport, String> {
    detect_impl(db, cfg, crops_dir, project_id, true)
}

fn detect_impl(
    db: &Db,
    cfg: &WorkerConfig,
    crops_dir: &Path,
    project_id: i64,
    force: bool,
) -> Result<DetectReport, String> {
    let assets = db.list_assets(project_id).map_err(|e| e.to_string())?;
    let mut session = WorkerSession::start(cfg, "face").map_err(|e| e.to_string())?;
    let (mut processed, mut found, mut failed, mut skipped) = (0usize, 0usize, 0usize, 0usize);
    for asset in &assets {
        if asset.detected && !force {
            skipped += 1;
            continue;
        }
        match session.call(json!({ "image": asset.path })) {
            Ok(resp) => {
                let detected = faces_from(&resp);
                let new_faces: Vec<_> = detected.into_iter().map(|(f, _)| f).collect();
                match db.replace_faces(asset.id, &new_faces) {
                    Ok(saved) => {
                        processed += 1;
                        found += saved.len();
                        let _ = db.clear_failure_for(asset.id, "detect");
                        let mut crop_error: Option<String> = None;
                        for face in &saved {
                            if let Err(e) = save_crop(
                                &asset.path,
                                face.bbox,
                                &crops_dir.join(format!("{}.jpg", face.id)),
                            ) {
                                if crop_error.is_none() {
                                    crop_error = Some(e);
                                }
                            }
                        }
                        match crop_error {
                            Some(e) => record(db, project_id, asset.id, "crop", &asset.path, &e),
                            None => {
                                let _ = db.clear_failure_for(asset.id, "crop");
                            }
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        record(
                            db,
                            project_id,
                            asset.id,
                            "detect",
                            &asset.path,
                            &e.to_string(),
                        );
                    }
                }
            }
            Err(e) => {
                failed += 1;
                record(
                    db,
                    project_id,
                    asset.id,
                    "detect",
                    &asset.path,
                    &e.to_string(),
                );
            }
        }
    }
    Ok(DetectReport {
        images_processed: processed,
        faces_found: found,
        failed,
        skipped,
    })
}

pub fn generate_embeddings(
    db: &Db,
    cfg: &WorkerConfig,
    project_id: i64,
) -> Result<EmbedReport, String> {
    let assets = db.list_assets(project_id).map_err(|e| e.to_string())?;
    let mut session = WorkerSession::start(cfg, "embedding").map_err(|e| e.to_string())?;
    let (mut embedded, mut failed, mut skipped) = (0usize, 0usize, 0usize);
    for asset in &assets {
        let stored = db.list_faces(asset.id).map_err(|e| e.to_string())?;
        if stored.is_empty() {
            continue;
        }
        // Skip faces that already have an embedding for this model
        // (W20): only assets with missing embeddings reach the worker.
        let done: std::collections::HashSet<i64> = db
            .embedded_face_ids(asset.id, EMBED_MODEL)
            .map_err(|e| e.to_string())?
            .into_iter()
            .collect();
        skipped += done.len();
        if stored.iter().all(|f| done.contains(&f.id)) {
            let _ = db.clear_failure_for(asset.id, "embed");
            continue;
        }
        match session.call(json!({ "image": asset.path })) {
            Ok(resp) => {
                let _ = db.clear_failure_for(asset.id, "embed");
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
                        if done.contains(&face.id) {
                            continue;
                        }
                        match db.save_embedding(face.id, EMBED_MODEL, &vector) {
                            Ok(()) => embedded += 1,
                            Err(e) => {
                                failed += 1;
                                record(
                                    db,
                                    project_id,
                                    asset.id,
                                    "embed",
                                    &asset.path,
                                    &e.to_string(),
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                failed += 1;
                record(
                    db,
                    project_id,
                    asset.id,
                    "embed",
                    &asset.path,
                    &e.to_string(),
                );
            }
        }
    }
    Ok(EmbedReport {
        embedded,
        failed,
        skipped,
    })
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
        // Guard (W7): with no embeddings there is nothing to cluster,
        // and an empty result must never wipe the identities the user
        // already has. Refuse with guidance instead of replacing.
        return Err("特徴量がまだ取得されていないため、人物を分類できません。\
             先に「特徴量を取得する」を実行してください\
             （登録済みの人物データはそのまま保持されています）"
            .into());
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

    // Ordered groups with their quality scores.
    let mut ordered: Vec<(Vec<i64>, f32)> = Vec::new();
    for group in &groups {
        let vectors: Vec<&Vec<f32>> = group
            .iter()
            .filter_map(|id| by_id.get(id).map(|f| &f.vector))
            .collect();
        ordered.push((group.clone(), mean_cosine_to_centroid(&vectors)));
    }
    for face_id in &noise {
        let quality = by_id.get(face_id).map(|f| f.confidence).unwrap_or(0.0);
        ordered.push((vec![*face_id], quality));
    }

    // Carry over (W5/W6): match the new groups to the previous
    // identities by face-set overlap. Matched groups keep the previous
    // identity's row (stable ID) and the name the user gave it;
    // unmatched groups are new people and get a fresh default name.
    let previous: Vec<PreviousIdentity> = db
        .identities_with_faces(project_id)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(identity, face_ids)| PreviousIdentity {
            id: identity.id,
            name: identity.name,
            face_ids,
        })
        .collect();
    let group_faces: Vec<Vec<i64>> = ordered.iter().map(|(faces, _)| faces.clone()).collect();
    let matches = match_previous(&previous, &group_faces);

    let mut names_carried_over = 0usize;
    let mut reconciled: Vec<ReconciledIdentity> = Vec::with_capacity(ordered.len());
    for (i, (face_ids, quality_score)) in ordered.into_iter().enumerate() {
        match matches[i] {
            Some(pi) => {
                names_carried_over += 1;
                reconciled.push(ReconciledIdentity {
                    keep_id: Some(previous[pi].id),
                    name: previous[pi].name.clone(),
                    quality_score,
                    face_ids,
                });
            }
            None => reconciled.push(ReconciledIdentity {
                keep_id: None,
                name: format!("人物 {}", i + 1),
                quality_score,
                face_ids,
            }),
        }
    }

    let total = reconciled.len();
    db.reconcile_identities(project_id, &reconciled)
        .map_err(|e| e.to_string())?;
    Ok(ClusterReport {
        identities: total,
        clustered_faces,
        single_faces,
        names_carried_over,
    })
}
