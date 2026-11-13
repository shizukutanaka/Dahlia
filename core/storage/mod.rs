//! SQLite storage layer. Owns the single database connection and all
//! SQL. Schema source of truth: database/schema.sql (embedded below).

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use serde::Serialize;

use crate::{Error, Result};

const SCHEMA: &str = include_str!("../../database/schema.sql");

#[derive(Debug, Clone, Serialize)]
pub struct Project {
    pub id: i64,
    pub name: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Asset {
    pub id: i64,
    pub project_id: i64,
    pub path: String,
    pub thumbnail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Face {
    pub id: i64,
    pub asset_id: i64,
    pub bbox: [f32; 4], // x, y, width, height (pixels)
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Identity {
    pub id: i64,
    pub project_id: i64,
    pub name: String,
    pub quality_score: f32,
    pub face_count: i64,
}

pub struct NewFace {
    pub bbox: [f32; 4],
    pub confidence: f32,
}

pub struct NewIdentity {
    pub name: String,
    pub quality_score: f32,
    pub face_ids: Vec<i64>,
}

/// One group of a fresh clustering result, optionally continuing a
/// previous identity (keep_id) so its row — and the name the user gave
/// it — survives re-clustering.
pub struct ReconciledIdentity {
    pub keep_id: Option<i64>,
    pub name: String,
    pub quality_score: f32,
    pub face_ids: Vec<i64>,
}

/// A face plus its embedding vector, used as clustering input.
pub struct FaceEmbedding {
    pub face_id: i64,
    pub asset_id: i64,
    pub confidence: f32,
    pub vector: Vec<f32>,
}

pub struct Db {
    conn: Mutex<Connection>,
    recovery_note: Option<String>,
}

/// Current database schema version, stored in PRAGMA user_version.
/// Version 1 is the original V0.1 schema (database/schema.sql). When
/// the schema changes, bump this and add a migration step in init().
const SCHEMA_VERSION: i64 = 1;

/// How many startup backups to keep next to the database file.
const MAX_BACKUPS: usize = 5;

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Protect an existing database before touching it: verify its
        // integrity, snapshot a backup, and recover from the newest
        // good backup when the file itself is corrupt — so a damaged
        // database never just makes the app fail to start.
        let mut recovery_note = None;
        if path.exists()
            && std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false)
        {
            recovery_note = Self::prepare_existing_file(path)?;
        }
        let conn = Connection::open(path)?;
        Self::init(&conn)?;
        Ok(Self { conn: Mutex::new(conn), recovery_note })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(&conn)?;
        Ok(Self { conn: Mutex::new(conn), recovery_note: None })
    }

    fn init(conn: &Connection) -> Result<()> {
        let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version > SCHEMA_VERSION {
            return Err(Error::Worker(format!(
                "このデータベースは新しいバージョンの Dahlia で作成されているため開けません\
                 （データの版 {version}、このアプリの対応版 {SCHEMA_VERSION}）"
            )));
        }
        conn.execute_batch(SCHEMA)?;
        if version < SCHEMA_VERSION {
            // Migration steps for future versions run here, oldest
            // first, before stamping the new version.
            conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?;
        }
        Ok(())
    }

    /// True when the database file at `path` passes SQLite's
    /// integrity check. Any open/query failure counts as not-ok.
    fn file_integrity_ok(path: &Path) -> bool {
        let Ok(conn) = Connection::open(path) else { return false };
        conn.query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
            .map(|first| first == "ok")
            .unwrap_or(false)
    }

    fn backups_dir(db_path: &Path) -> Option<std::path::PathBuf> {
        db_path.parent().map(|p| p.join("backups"))
    }

    /// Snapshot the (verified healthy) database into backups/ and
    /// prune old generations. Best effort: a failed backup must not
    /// prevent the app from starting.
    fn create_backup(db_path: &Path) {
        let Some(dir) = Self::backups_dir(db_path) else { return };
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let secs = unix_secs();
        let mut dest = dir.join(format!("dahlia-{secs}.db"));
        let mut n = 1;
        while dest.exists() {
            dest = dir.join(format!("dahlia-{secs}-{n}.db"));
            n += 1;
        }
        let Ok(conn) = Connection::open(db_path) else { return };
        // VACUUM INTO writes a consistent copy even if the source is
        // mid-journal; the path is app-controlled, quotes escaped.
        let sql = format!(
            "VACUUM INTO '{}'",
            dest.to_string_lossy().replace('\'', "''")
        );
        if conn.execute_batch(&sql).is_err() {
            return;
        }
        // Rotate: keep the newest MAX_BACKUPS generations.
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.starts_with("dahlia-") && n.ends_with(".db"))
                            .unwrap_or(false)
                    })
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        while files.len() > MAX_BACKUPS {
            let _ = std::fs::remove_file(files.remove(0));
        }
    }

    fn newest_valid_backup(db_path: &Path) -> Option<std::path::PathBuf> {
        let dir = Self::backups_dir(db_path)?;
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("dahlia-") && n.ends_with(".db"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        files.iter().rev().find(|p| Self::file_integrity_ok(p)).cloned()
    }

    /// Verify an existing database file, back it up, or recover it.
    /// Returns a user-facing note when a recovery happened.
    fn prepare_existing_file(path: &Path) -> Result<Option<String>> {
        if Self::file_integrity_ok(path) {
            Self::create_backup(path);
            return Ok(None);
        }
        // The database is corrupt. Restore the newest backup that
        // itself passes the integrity check, keeping the corrupt file
        // for inspection instead of silently discarding it.
        let Some(backup) = Self::newest_valid_backup(path) else {
            return Err(Error::Worker(
                "データベースが破損しており、復旧に使えるバックアップがありません。\
                 データフォルダを確認してください"
                    .into(),
            ));
        };
        let preserved = path.with_extension(format!("corrupt-{}.db", unix_secs()));
        let _ = std::fs::rename(path, &preserved);
        for suffix in ["-journal", "-wal", "-shm"] {
            let side = format!("{}{}", path.to_string_lossy(), suffix);
            let _ = std::fs::remove_file(side);
        }
        std::fs::copy(&backup, path)?;
        let name = backup
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(Some(format!(
            "データベースが破損していたため、バックアップ（{name}）から復旧しました。\
             破損したファイルは {} として残してあります",
            preserved.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
        )))
    }

    /// User-facing note when startup recovered the database from a
    /// backup (None in the normal case).
    pub fn recovery_note(&self) -> Option<&str> {
        self.recovery_note.as_deref()
    }

    /// Run SQLite's integrity check on the live database.
    pub fn integrity_ok(&self) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let first: String =
            conn.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
        Ok(first == "ok")
    }

    /// The schema version stamped on the database (PRAGMA user_version).
    pub fn schema_version(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?)
    }

    /// Names of user tables present (used by the db_status command).
    pub fn table_names(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' \
             AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ---------------- projects ----------------

    pub fn create_project(&self, name: &str) -> Result<Project> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Worker("project name is empty".into()));
        }
        let conn = self.conn.lock().unwrap();
        conn.execute("INSERT INTO projects (name) VALUES (?1)", params![name])?;
        let id = conn.last_insert_rowid();
        Ok(conn.query_row(
            "SELECT id, name, created_at FROM projects WHERE id = ?1",
            params![id],
            |r| Ok(Project { id: r.get(0)?, name: r.get(1)?, created_at: r.get(2)? }),
        )?)
    }

    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, created_at FROM projects ORDER BY created_at, id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Project { id: r.get(0)?, name: r.get(1)?, created_at: r.get(2)? })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ---------------- assets ----------------

    /// Insert an asset; a duplicate (project_id, path) returns the
    /// existing row instead of an error, so re-importing a folder is safe.
    pub fn add_asset(&self, project_id: i64, path: &str, thumbnail: Option<&str>) -> Result<Asset> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO assets (project_id, path, thumbnail) VALUES (?1, ?2, ?3)",
            params![project_id, path, thumbnail],
        )?;
        Ok(conn.query_row(
            "SELECT id, project_id, path, thumbnail FROM assets WHERE project_id = ?1 AND path = ?2",
            params![project_id, path],
            |r| Ok(Asset { id: r.get(0)?, project_id: r.get(1)?, path: r.get(2)?, thumbnail: r.get(3)? }),
        )?)
    }

    pub fn set_asset_thumbnail(&self, asset_id: i64, thumbnail: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE assets SET thumbnail = ?1 WHERE id = ?2",
            params![thumbnail, asset_id],
        )?;
        Ok(())
    }

    pub fn list_assets(&self, project_id: i64) -> Result<Vec<Asset>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, project_id, path, thumbnail FROM assets WHERE project_id = ?1 ORDER BY path",
        )?;
        let rows = stmt.query_map(params![project_id], |r| {
            Ok(Asset { id: r.get(0)?, project_id: r.get(1)?, path: r.get(2)?, thumbnail: r.get(3)? })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_asset(&self, asset_id: i64) -> Result<Asset> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row(
            "SELECT id, project_id, path, thumbnail FROM assets WHERE id = ?1",
            params![asset_id],
            |r| Ok(Asset { id: r.get(0)?, project_id: r.get(1)?, path: r.get(2)?, thumbnail: r.get(3)? }),
        ).map_err(|_| Error::NotFound(format!("asset {asset_id}")))?)
    }

    // ---------------- faces & embeddings ----------------

    /// Replace all faces of an asset with fresh detection results.
    pub fn replace_faces(&self, asset_id: i64, faces: &[NewFace]) -> Result<Vec<Face>> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM faces WHERE asset_id = ?1", params![asset_id])?;
        let mut out = Vec::new();
        for f in faces {
            let bbox = serde_json::to_string(&f.bbox)?;
            tx.execute(
                "INSERT INTO faces (asset_id, bbox, confidence) VALUES (?1, ?2, ?3)",
                params![asset_id, bbox, f.confidence],
            )?;
            out.push(Face {
                id: tx.last_insert_rowid(),
                asset_id,
                bbox: f.bbox,
                confidence: f.confidence,
            });
        }
        tx.commit()?;
        Ok(out)
    }

    pub fn list_faces(&self, asset_id: i64) -> Result<Vec<Face>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, asset_id, bbox, confidence FROM faces WHERE asset_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(params![asset_id], |r| {
            let bbox_s: String = r.get(2)?;
            let bbox: [f32; 4] = serde_json::from_str(&bbox_s).unwrap_or([0.0; 4]);
            Ok(Face { id: r.get(0)?, asset_id: r.get(1)?, bbox, confidence: r.get(3)? })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn save_embedding(&self, face_id: i64, model: &str, vector: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for v in vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO embeddings (face_id, model, vector) VALUES (?1, ?2, ?3) \
             ON CONFLICT(face_id, model) DO UPDATE SET vector = excluded.vector",
            params![face_id, model, bytes],
        )?;
        Ok(())
    }

    pub fn faces_with_embeddings(&self, project_id: i64, model: &str) -> Result<Vec<FaceEmbedding>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT f.id, f.asset_id, f.confidence, e.vector \
             FROM faces f \
             JOIN assets a ON a.id = f.asset_id \
             JOIN embeddings e ON e.face_id = f.id AND e.model = ?2 \
             WHERE a.project_id = ?1 ORDER BY f.id",
        )?;
        let rows = stmt.query_map(params![project_id, model], |r| {
            let blob: Vec<u8> = r.get(3)?;
            let vector = blob
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(FaceEmbedding { face_id: r.get(0)?, asset_id: r.get(1)?, confidence: r.get(2)?, vector })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ---------------- identities ----------------

    /// Replace all identities of a project with a fresh clustering result.
    pub fn replace_identities(&self, project_id: i64, items: &[NewIdentity]) -> Result<Vec<Identity>> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM identities WHERE project_id = ?1", params![project_id])?;
        let mut out = Vec::new();
        for item in items {
            tx.execute(
                "INSERT INTO identities (project_id, name, quality_score) VALUES (?1, ?2, ?3)",
                params![project_id, item.name, item.quality_score],
            )?;
            let id = tx.last_insert_rowid();
            for face_id in &item.face_ids {
                tx.execute(
                    "INSERT OR IGNORE INTO identity_faces (identity_id, face_id) VALUES (?1, ?2)",
                    params![id, face_id],
                )?;
            }
            out.push(Identity {
                id,
                project_id,
                name: item.name.clone(),
                quality_score: item.quality_score,
                face_count: item.face_ids.len() as i64,
            });
        }
        tx.commit()?;
        Ok(out)
    }

    /// All identities of a project with their face IDs — the snapshot
    /// used to carry names/IDs over to a fresh clustering result.
    pub fn identities_with_faces(&self, project_id: i64) -> Result<Vec<(Identity, Vec<i64>)>> {
        let identities = self.list_identities(project_id)?;
        let mut out = Vec::with_capacity(identities.len());
        for identity in identities {
            let face_ids = self.identity_face_ids(identity.id)?;
            out.push((identity, face_ids));
        }
        Ok(out)
    }

    /// Apply a fresh clustering result, preserving matched identities.
    ///
    /// Items with `keep_id` update that existing row in place (stable
    /// ID; the caller supplies the carried-over name), unmatched
    /// previous identities are deleted, and items without `keep_id`
    /// are inserted as new identities. Refuses to wipe existing
    /// identities with an empty result — an empty clustering run must
    /// never silently erase the people the user has named.
    pub fn reconcile_identities(
        &self,
        project_id: i64,
        items: &[ReconciledIdentity],
    ) -> Result<Vec<Identity>> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let mut stmt =
            tx.prepare("SELECT id FROM identities WHERE project_id = ?1")?;
        let existing_ids: Vec<i64> = stmt
            .query_map(params![project_id], |r| r.get(0))?
            .collect::<std::result::Result<_, _>>()?;
        drop(stmt);

        if items.is_empty() && !existing_ids.is_empty() {
            return Err(Error::Worker(
                "refusing to replace existing identities with an empty clustering result"
                    .into(),
            ));
        }
        for item in items {
            if let Some(keep) = item.keep_id {
                if !existing_ids.contains(&keep) {
                    return Err(Error::Worker(format!(
                        "identity {keep} does not belong to project {project_id}"
                    )));
                }
            }
        }

        tx.execute(
            "DELETE FROM identity_faces WHERE identity_id IN \
             (SELECT id FROM identities WHERE project_id = ?1)",
            params![project_id],
        )?;
        let keep_ids: Vec<i64> = items.iter().filter_map(|i| i.keep_id).collect();
        for id in &existing_ids {
            if !keep_ids.contains(id) {
                tx.execute("DELETE FROM identities WHERE id = ?1", params![id])?;
            }
        }

        let mut final_ids: Vec<i64> = Vec::with_capacity(items.len());
        for item in items {
            let id = match item.keep_id {
                Some(keep) => {
                    tx.execute(
                        "UPDATE identities SET name = ?1, quality_score = ?2 WHERE id = ?3",
                        params![item.name, item.quality_score, keep],
                    )?;
                    keep
                }
                None => {
                    tx.execute(
                        "INSERT INTO identities (project_id, name, quality_score) \
                         VALUES (?1, ?2, ?3)",
                        params![project_id, item.name, item.quality_score],
                    )?;
                    tx.last_insert_rowid()
                }
            };
            for face_id in &item.face_ids {
                tx.execute(
                    "INSERT OR IGNORE INTO identity_faces (identity_id, face_id) \
                     VALUES (?1, ?2)",
                    params![id, face_id],
                )?;
            }
            final_ids.push(id);
        }

        let mut out = Vec::with_capacity(final_ids.len());
        for id in final_ids {
            let identity = tx.query_row(
                "SELECT i.id, i.project_id, i.name, i.quality_score, COUNT(f.face_id) \
                 FROM identities i LEFT JOIN identity_faces f ON f.identity_id = i.id \
                 WHERE i.id = ?1 GROUP BY i.id",
                params![id],
                |r| {
                    Ok(Identity {
                        id: r.get(0)?,
                        project_id: r.get(1)?,
                        name: r.get(2)?,
                        quality_score: r.get(3)?,
                        face_count: r.get(4)?,
                    })
                },
            )?;
            out.push(identity);
        }
        tx.commit()?;
        Ok(out)
    }

    pub fn list_identities(&self, project_id: i64) -> Result<Vec<Identity>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT i.id, i.project_id, i.name, i.quality_score, COUNT(f.face_id) \
             FROM identities i LEFT JOIN identity_faces f ON f.identity_id = i.id \
             WHERE i.project_id = ?1 GROUP BY i.id ORDER BY i.quality_score DESC, i.id",
        )?;
        let rows = stmt.query_map(params![project_id], |r| {
            Ok(Identity { id: r.get(0)?, project_id: r.get(1)?, name: r.get(2)?, quality_score: r.get(3)?, face_count: r.get(4)? })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn identity_face_ids(&self, identity_id: i64) -> Result<Vec<i64>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT face_id FROM identity_faces WHERE identity_id = ?1 ORDER BY face_id",
        )?;
        let rows = stmt.query_map(params![identity_id], |r| r.get::<_, i64>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_identity(&self, identity_id: i64) -> Result<Identity> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row(
            "SELECT i.id, i.project_id, i.name, i.quality_score, COUNT(f.face_id) \
             FROM identities i LEFT JOIN identity_faces f ON f.identity_id = i.id \
             WHERE i.id = ?1 GROUP BY i.id",
            params![identity_id],
            |r| Ok(Identity { id: r.get(0)?, project_id: r.get(1)?, name: r.get(2)?, quality_score: r.get(3)?, face_count: r.get(4)? }),
        ).map_err(|_| Error::NotFound(format!("identity {identity_id}")))?)
    }

    pub fn rename_identity(&self, identity_id: i64, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Worker("identity name is empty".into()));
        }
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE identities SET name = ?1 WHERE id = ?2",
            params![name, identity_id],
        )?;
        Ok(())
    }

    /// Faces of an identity joined with their asset (for the detail UI).
    pub fn identity_faces_detailed(&self, identity_id: i64) -> Result<Vec<(Face, Asset)>> {
        let face_ids = self.identity_face_ids(identity_id)?;
        let mut out = Vec::new();
        for fid in face_ids {
            // Scope the lock: get_asset() below locks the same Mutex,
            // so holding this guard across the call self-deadlocks.
            let face = {
                let conn = self.conn.lock().unwrap();
                conn.query_row(
                    "SELECT id, asset_id, bbox, confidence FROM faces WHERE id = ?1",
                    params![fid],
                    |r| {
                        let bbox_s: String = r.get(2)?;
                        let bbox: [f32; 4] = serde_json::from_str(&bbox_s).unwrap_or([0.0; 4]);
                        Ok(Face { id: r.get(0)?, asset_id: r.get(1)?, bbox, confidence: r.get(3)? })
                    },
                )
            };
            if let Ok(face) = face {
                let asset = self.get_asset(face.asset_id)?;
                out.push((face, asset));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_creates_all_tables() {
        let db = Db::open_in_memory().unwrap();
        let tables = db.table_names().unwrap();
        for t in ["projects", "assets", "faces", "embeddings", "identities", "identity_faces"] {
            assert!(tables.contains(&t.to_string()), "missing table {t}");
        }
    }

    #[test]
    fn project_crud_roundtrip() {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project("  Alice  ").unwrap();
        assert_eq!(p.name, "Alice");
        let all = db.list_projects().unwrap();
        assert_eq!(all.len(), 1);
        assert!(db.create_project("   ").is_err());
    }

    #[test]
    fn asset_dedup_and_faces() {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project("P").unwrap();
        let a1 = db.add_asset(p.id, "/photos/a.jpg", None).unwrap();
        let a2 = db.add_asset(p.id, "/photos/a.jpg", None).unwrap();
        assert_eq!(a1.id, a2.id);
        let faces = db
            .replace_faces(a1.id, &[NewFace { bbox: [1.0, 2.0, 30.0, 40.0], confidence: 0.9 }])
            .unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(db.list_faces(a1.id).unwrap()[0].bbox, [1.0, 2.0, 30.0, 40.0]);
    }

    #[test]
    fn embedding_and_identity_roundtrip() {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project("P").unwrap();
        let a = db.add_asset(p.id, "/x.jpg", None).unwrap();
        let f = db.replace_faces(a.id, &[NewFace { bbox: [0.0; 4], confidence: 0.8 }]).unwrap();
        db.save_embedding(f[0].id, "arcface-test", &[0.1, 0.2, 0.3]).unwrap();
        let fes = db.faces_with_embeddings(p.id, "arcface-test").unwrap();
        assert_eq!(fes.len(), 1);
        assert_eq!(fes[0].vector, vec![0.1, 0.2, 0.3]);
        let ids = db
            .replace_identities(p.id, &[NewIdentity {
                name: "人物 1".into(),
                quality_score: 0.75,
                face_ids: vec![f[0].id],
            }])
            .unwrap();
        assert_eq!(ids[0].face_count, 1);
        db.rename_identity(ids[0].id, "Alice").unwrap();
        assert_eq!(db.list_identities(p.id).unwrap()[0].name, "Alice");
    }

    fn fresh_faces(db: &Db, p: i64, n: usize) -> Vec<i64> {
        let a = db.add_asset(p, "/x.jpg", None).unwrap();
        let faces = db
            .replace_faces(
                a.id,
                &(0..n)
                    .map(|_| NewFace { bbox: [0.0; 4], confidence: 0.9 })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        faces.iter().map(|f| f.id).collect()
    }

    #[test]
    fn reconcile_keeps_matched_identity_id_and_name() {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project("P").unwrap();
        let f = fresh_faces(&db, p.id, 3);
        let ids = db
            .replace_identities(p.id, &[NewIdentity {
                name: "人物 1".into(),
                quality_score: 0.5,
                face_ids: f.clone(),
            }])
            .unwrap();
        db.rename_identity(ids[0].id, "Alice").unwrap();

        // Re-cluster: Alice's group (same faces) plus a new person.
        let out = db
            .reconcile_identities(p.id, &[
                ReconciledIdentity {
                    keep_id: Some(ids[0].id),
                    name: "Alice".into(),
                    quality_score: 0.8,
                    face_ids: f.clone(),
                },
                ReconciledIdentity {
                    keep_id: None,
                    name: "人物 2".into(),
                    quality_score: 0.3,
                    face_ids: vec![],
                },
            ])
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, ids[0].id, "matched identity keeps its row id");
        assert_eq!(out[0].name, "Alice");
        assert_eq!(out[0].face_count, 3);
        let listed = db.list_identities(p.id).unwrap();
        assert_eq!(listed.iter().filter(|i| i.name == "Alice").count(), 1);
    }

    #[test]
    fn reconcile_deletes_unmatched_previous_identities() {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project("P").unwrap();
        let f = fresh_faces(&db, p.id, 2);
        db.replace_identities(p.id, &[
            NewIdentity { name: "A".into(), quality_score: 0.5, face_ids: vec![f[0]] },
            NewIdentity { name: "B".into(), quality_score: 0.5, face_ids: vec![f[1]] },
        ])
        .unwrap();
        let out = db
            .reconcile_identities(p.id, &[ReconciledIdentity {
                keep_id: None,
                name: "C".into(),
                quality_score: 0.5,
                face_ids: f.clone(),
            }])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(db.list_identities(p.id).unwrap().len(), 1);
    }

    #[test]
    fn reconcile_refuses_to_wipe_identities_with_empty_result() {
        let db = Db::open_in_memory().unwrap();
        let p = db.create_project("P").unwrap();
        let f = fresh_faces(&db, p.id, 1);
        db.replace_identities(p.id, &[NewIdentity {
            name: "Alice".into(),
            quality_score: 0.5,
            face_ids: f,
        }])
        .unwrap();
        assert!(db.reconcile_identities(p.id, &[]).is_err());
        assert_eq!(db.list_identities(p.id).unwrap().len(), 1);
    }

    fn temp_db_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!("dahlia-test-{}-{}", std::process::id(), tag))
            .join("dahlia.db")
    }

    #[test]
    fn schema_version_is_stamped_and_checked() {
        let path = temp_db_path("version");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        {
            let db = Db::open(&path).unwrap();
            assert_eq!(db.schema_version().unwrap(), SCHEMA_VERSION);
            assert!(db.integrity_ok().unwrap());
            assert!(db.recovery_note().is_none());
        }
        // A database stamped by a newer app version must be refused.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch("PRAGMA user_version = 99").unwrap();
        }
        assert!(Db::open(&path).is_err());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn reopen_creates_a_backup() {
        let path = temp_db_path("backup");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        {
            let db = Db::open(&path).unwrap();
            db.create_project("Persist").unwrap();
        }
        {
            let db = Db::open(&path).unwrap();
            assert_eq!(db.list_projects().unwrap().len(), 1);
        }
        let backups = path.parent().unwrap().join("backups");
        let count = std::fs::read_dir(&backups)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".db"))
            .count();
        assert_eq!(count, 1, "one startup backup expected");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn corrupt_database_is_recovered_from_backup() {
        let path = temp_db_path("corrupt");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        {
            let db = Db::open(&path).unwrap();
            db.create_project("KeepMe").unwrap();
        }
        // Second open snapshots the healthy database into backups/.
        {
            let _db = Db::open(&path).unwrap();
        }
        std::fs::write(&path, b"this is not a sqlite database at all").unwrap();
        let db = Db::open(&path).unwrap();
        assert!(
            db.recovery_note().is_some(),
            "recovery must be reported to the user"
        );
        let projects = db.list_projects().unwrap();
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].name, "KeepMe");
        // The corrupt file is preserved for inspection.
        let preserved = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().contains("corrupt-"));
        assert!(preserved);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn data_survives_reopen() {
        let dir = std::env::temp_dir().join(format!("dahlia-test-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        {
            let db = Db::open(&dir).unwrap();
            db.create_project("Persist").unwrap();
        }
        let db = Db::open(&dir).unwrap();
        assert_eq!(db.list_projects().unwrap().len(), 1);
        let _ = std::fs::remove_file(&dir);
    }
}
