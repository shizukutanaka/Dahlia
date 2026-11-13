//! Folder import: scan a directory tree for images, register them as
//! assets and generate thumbnails into the app data directory.

use std::path::{Path, PathBuf};

use dahlia_core::storage::{Asset, Db};
use serde::Serialize;

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif"];

#[derive(Serialize)]
pub struct ImportReport {
    pub added: usize,
    pub already_present: usize,
    pub failed: usize,
    pub assets: Vec<Asset>,
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
    thumbs_dir: &Path,
    project_id: i64,
    folder: &Path,
) -> Result<ImportReport, String> {
    let mut files = Vec::new();
    collect_images(folder, &mut files);
    files.sort();
    let existing: std::collections::HashSet<String> = db
        .list_assets(project_id)
        .map(|v| v.into_iter().map(|a| a.path).collect())
        .unwrap_or_default();
    let (mut added, mut present, mut failed) = (0usize, 0usize, 0usize);
    for file in files {
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
                continue;
            }
        };
        if is_new {
            added += 1;
        } else {
            present += 1;
        }
        if asset.thumbnail.is_none() {
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
    }
    let assets = db.list_assets(project_id).map_err(|e| e.to_string())?;
    Ok(ImportReport {
        added,
        already_present: present,
        failed,
        assets,
    })
}
