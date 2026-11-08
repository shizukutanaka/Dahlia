-- Dahlia V0.1 schema (single source of truth; applied by core/storage)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS assets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    thumbnail  TEXT,
    UNIQUE(project_id, path)
);

CREATE TABLE IF NOT EXISTS faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id   INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    bbox       TEXT NOT NULL,  -- JSON: [x, y, width, height] in pixels
    confidence REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    model   TEXT NOT NULL,
    vector  BLOB NOT NULL,     -- little-endian f32 array
    UNIQUE(face_id, model)
);

CREATE TABLE IF NOT EXISTS identities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    quality_score REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS identity_faces (
    identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    face_id     INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    PRIMARY KEY (identity_id, face_id)
);

CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id);
CREATE INDEX IF NOT EXISTS idx_faces_asset ON faces(asset_id);
