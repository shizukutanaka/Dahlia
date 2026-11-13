//! HEIC/HEIF photos (W87): the iPhone default format.
//!
//! The Python workers decode HEIC through Pillow (pillow-heif), but
//! the Rust side reads pixels with the `image` crate — thumbnails
//! and face crops — which has no HEIC decoder. So each HEIC photo
//! gets a full-resolution JPEG working copy (a "proxy") in the app
//! data directory, written by workers/heif_convert.py through the
//! same loader the workers use: the proxy's pixels are exactly the
//! pixels detection saw (orientation already applied, no EXIF tag
//! re-written), and every Rust-side read of a HEIC photo goes
//! through the proxy instead of the original file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use dahlia_core::jobs::WorkerConfig;
use serde::Deserialize;

pub fn is_heif_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("heic") || e.eq_ignore_ascii_case("heif"))
        .unwrap_or(false)
}

/// The proxy location of an asset: deterministic, so no database
/// column is needed to find it again.
pub fn proxy_path(proxies_dir: &Path, asset_id: i64) -> PathBuf {
    proxies_dir.join(format!("{asset_id}.jpg"))
}

/// The file Rust-side pixel reads should use for an asset: its
/// proxy when one exists, otherwise the original path.
pub fn raster_path(proxies_dir: &Path, asset_id: i64, original: &Path) -> PathBuf {
    let proxy = proxy_path(proxies_dir, asset_id);
    if proxy.exists() {
        proxy
    } else {
        original.to_path_buf()
    }
}

#[derive(Deserialize)]
struct ConvertLine {
    src: Option<String>,
    ok: bool,
    #[serde(default)]
    error: Option<String>,
}

/// Make sure every listed asset has a proxy, converting the missing
/// ones in a single Python invocation. Assets whose proxy already
/// exists are reported as ready without invoking anything. One
/// result per input asset, in input order.
pub fn ensure_proxies(
    cfg: &WorkerConfig,
    proxies_dir: &Path,
    items: &[(i64, PathBuf)],
) -> Vec<(i64, Result<PathBuf, String>)> {
    let mut results: HashMap<i64, Result<PathBuf, String>> = HashMap::new();
    let mut missing: Vec<(i64, PathBuf, PathBuf)> = Vec::new(); // (id, src, dst)
    for (id, src) in items {
        let dst = proxy_path(proxies_dir, *id);
        if dst.exists() {
            results.insert(*id, Ok(dst));
        } else {
            missing.push((*id, src.clone(), dst));
        }
    }
    let in_order = |results: &mut HashMap<i64, Result<PathBuf, String>>| {
        items
            .iter()
            .map(|(id, _)| (*id, results.remove(id).unwrap()))
            .collect::<Vec<_>>()
    };
    if missing.is_empty() {
        return in_order(&mut results);
    }
    if let Err(e) = std::fs::create_dir_all(proxies_dir) {
        for (id, _, _) in &missing {
            results.insert(*id, Err(e.to_string()));
        }
        return in_order(&mut results);
    }

    let script = cfg.workers_dir.join("heif_convert.py");
    let mut cmd = Command::new(&cfg.python);
    cmd.arg("-u").arg(&script);
    for (_, src, dst) in &missing {
        cmd.arg(src).arg(dst);
    }
    let converted: Result<HashMap<String, Result<(), String>>, String> = (|| {
        let out = cmd
            .output()
            .map_err(|e| format!("HEIC 変換プログラムを起動できませんでした: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut by_src: HashMap<String, Result<(), String>> = HashMap::new();
        for line in stdout.lines() {
            if let Ok(parsed) = serde_json::from_str::<ConvertLine>(line) {
                if let Some(src) = parsed.src {
                    by_src.insert(
                        src,
                        if parsed.ok {
                            Ok(())
                        } else {
                            Err(parsed
                                .error
                                .unwrap_or_else(|| "HEIC 変換に失敗しました".to_string()))
                        },
                    );
                }
            }
        }
        if by_src.is_empty() {
            // The script prints one line per pair; silence means it
            // died before converting anything (missing component,
            // syntax error, ...). Surface its stderr instead.
            let stderr = String::from_utf8_lossy(&out.stderr);
            let detail = stderr.lines().last().unwrap_or("").trim();
            return Err(if detail.is_empty() {
                "HEIC 変換プログラムが応答しませんでした".to_string()
            } else {
                format!("HEIC 変換に失敗しました: {detail}")
            });
        }
        Ok(by_src)
    })();

    match converted {
        Ok(by_src) => {
            for (id, src, dst) in &missing {
                let key = src.to_string_lossy().into_owned();
                let result = match by_src.get(&key) {
                    Some(Ok(())) if dst.exists() => Ok(dst.clone()),
                    Some(Ok(())) => Err("HEIC 変換後のファイルが見つかりません".to_string()),
                    Some(Err(e)) => Err(e.clone()),
                    None => Err("HEIC 変換の結果が返りませんでした".to_string()),
                };
                results.insert(*id, result);
            }
        }
        Err(e) => {
            for (id, _, _) in &missing {
                results.insert(*id, Err(e.clone()));
            }
        }
    }
    in_order(&mut results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heif_extensions_match_case_insensitively() {
        assert!(is_heif_path(Path::new("a/b/photo.heic")));
        assert!(is_heif_path(Path::new("photo.HEIF")));
        assert!(is_heif_path(Path::new("photo.Heic")));
        assert!(!is_heif_path(Path::new("photo.jpg")));
        assert!(!is_heif_path(Path::new("photo.heic.jpg")));
        assert!(!is_heif_path(Path::new("noext")));
    }

    #[test]
    fn proxy_path_is_deterministic_per_asset() {
        let dir = Path::new("/data/proxies");
        assert_eq!(proxy_path(dir, 42), Path::new("/data/proxies/42.jpg"));
        assert_eq!(proxy_path(dir, 7), Path::new("/data/proxies/7.jpg"));
    }

    #[test]
    fn raster_path_falls_back_to_original_without_proxy() {
        let dir = std::env::temp_dir().join("dahlia-heif-test-no-proxies");
        let original = Path::new("/photos/img.heic");
        assert_eq!(raster_path(&dir, 99, original), original.to_path_buf());
    }
}
