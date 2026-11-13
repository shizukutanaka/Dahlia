//! Folder import: scan a directory tree for images, register them as
//! assets and generate thumbnails into the app data directory.

use std::path::{Path, PathBuf};

use dahlia_core::jobs::{JobHooks, WorkerConfig};
use dahlia_core::storage::{Asset, Db};
use serde::Serialize;

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif", "heic", "heif"];

#[derive(Serialize)]
pub struct ImportReport {
    pub added: usize,
    pub already_present: usize,
    pub failed: usize,
    pub assets: Vec<Asset>,
    /// The user stopped the import partway; photos registered
    /// before that point stay registered.
    pub cancelled: bool,
}

fn collect_images(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_images(&path, out);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if IMAGE_EXTS.contains(&ext.to_lowercase().as_str()) {
                out.push(path);
            }
        }
    }
}

fn make_thumbnail(src: &Path, dest: &Path) -> Result<(), String> {
    let img = crate::open_oriented_image(src)?;
    let thumb = img.thumbnail(320, 320);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    thumb
        .save_with_format(dest, image::ImageFormat::Jpeg)
        .map_err(|e| e.to_string())
}

pub fn import_folder(
    db: &Db,
    cfg: &WorkerConfig,
    thumbs_dir: &Path,
    proxies_dir: &Path,
    project_id: i64,
    folder: &Path,
) -> Result<ImportReport, String> {
    import_folder_hooked(
        db,
        cfg,
        thumbs_dir,
        proxies_dir,
        project_id,
        folder,
        &JobHooks::none(),
    )
}

pub fn import_folder_hooked(
    db: &Db,
    cfg: &WorkerConfig,
    thumbs_dir: &Path,
    proxies_dir: &Path,
    project_id: i64,
    folder: &Path,
    hooks: &JobHooks,
) -> Result<ImportReport, String> {
    let mut files = Vec::new();
    collect_images(folder, &mut files);
    files.sort();
    let existing: std::collections::HashSet<String> = db
        .list_assets(project_id)
        .map(|v| v.into_iter().map(|a| a.path).collect())
        .unwrap_or_default();
    let (mut added, mut present, mut failed) = (0usize, 0usize, 0usize);
    let mut cancelled = false;
    // HEIC photos whose thumbnails must wait for a JPEG proxy
    // (the image crate cannot decode HEIC itself, W87). Converted
    // in one batch after the registration loop.
    let mut pending_heif: Vec<(i64, PathBuf)> = Vec::new();
    let total = files.len() as u64;
    hooks.report(0, total, None);
    for (i, file) in files.into_iter().enumerate() {
        if hooks.is_cancelled() {
            cancelled = true;
            break;
        }
        let path_str = file.to_string_lossy().to_string();
        let is_new = !existing.contains(&path_str);
        let asset = match db.add_asset(project_id, &path_str, None) {
            Ok(a) => a,
            Err(e) => {
                failed += 1;
                // Keep the failure visible (W11): which file, why.
                let _ = db.record_failure(
                    project_id,
                    None,
                    "import",
                    &path_str,
                    &crate::errors::friendly(&e.to_string()),
                );
                hooks.report(i as u64 + 1, total, None);
                continue;
            }
        };
        if is_new {
            added += 1;
        } else {
            present += 1;
        }
        if asset.thumbnail.is_none() && crate::heif::is_heif_path(&file) {
            pending_heif.push((asset.id, file.clone()));
        } else if asset.thumbnail.is_none() {
            let dest = thumbs_dir.join(format!("{}.jpg", asset.id));
            match make_thumbnail(&file, &dest) {
                Ok(()) => {
                    let _ = db.set_asset_thumbnail(asset.id, &dest.to_string_lossy());
                    let _ = db.clear_failure_for(asset.id, "import");
                }
                Err(e) => {
                    failed += 1;
                    let _ = db.record_failure(
                        project_id,
                        Some(asset.id),
                        "import",
                        &path_str,
                        &crate::errors::friendly(&e),
                    );
                }
            }
        }
        hooks.report(i as u64 + 1, total, None);
    }
    // HEIC proxies + thumbnails. Skipped when the user cancelled:
    // the assets stay registered without thumbnails, and the next
    // import of the same folder picks them up again here.
    if !cancelled && !pending_heif.is_empty() {
        let pending_total = pending_heif.len() as u64;
        hooks.report(0, pending_total, Some("HEIC変換"));
        let results = crate::heif::ensure_proxies(cfg, proxies_dir, &pending_heif);
        for (done, ((asset_id, file), (_, result))) in
            pending_heif.iter().zip(results.iter()).enumerate()
        {
            let path_str = file.to_string_lossy().to_string();
            let dest = thumbs_dir.join(format!("{asset_id}.jpg"));
            let thumb_result = match result {
                Ok(proxy) => make_thumbnail(proxy, &dest),
                Err(conv_err) => Err(conv_err.clone()),
            };
            match thumb_result {
                Ok(()) => {
                    let _ = db.set_asset_thumbnail(*asset_id, &dest.to_string_lossy());
                    let _ = db.clear_failure_for(*asset_id, "import");
                }
                Err(e) => {
                    failed += 1;
                    let _ = db.record_failure(
                        project_id,
                        Some(*asset_id),
                        "import",
                        &path_str,
                        &crate::errors::friendly(&e),
                    );
                }
            }
            hooks.report(done as u64 + 1, pending_total, Some("HEIC変換"));
        }
    }
    let assets = db.list_assets(project_id).map_err(|e| e.to_string())?;
    Ok(ImportReport {
        added,
        already_present: present,
        failed,
        assets,
        cancelled,
    })
}
