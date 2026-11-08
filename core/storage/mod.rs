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

/// A face plus its embedding vector, used as clustering input.
pub struct FaceEmbedding {
    pub face_id: i64,
    pub asset_id: i64,
    pub confidence: f32,
    pub vector: Vec<f32>,
}

pub struct Db {
    conn: Mutex<Connection>,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Self::init(&conn)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(&conn)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    fn init(conn: &Connection) -> Result<()> {
        conn.execute_batch(SCHEMA)?;
        Ok(())
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
            let conn = self.conn.lock().unwrap();
            let face = conn.query_row(
                "SELECT id, asset_id, bbox, confidence FROM faces WHERE id = ?1",
                params![fid],
                |r| {
                    let bbox_s: String = r.get(2)?;
                    let bbox: [f32; 4] = serde_json::from_str(&bbox_s).unwrap_or([0.0; 4]);
                    Ok(Face { id: r.get(0)?, asset_id: r.get(1)?, bbox, confidence: r.get(3)? })
                },
            );
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
