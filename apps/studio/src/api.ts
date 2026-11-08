import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export interface Project {
  id: number;
  name: string;
  created_at: string;
}

export interface Asset {
  id: number;
  project_id: number;
  path: string;
  thumbnail: string | null;
}

export interface Face {
  id: number;
  asset_id: number;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface Identity {
  id: number;
  project_id: number;
  name: string;
  quality_score: number;
  face_count: number;
}

export interface IdentityDetail {
  identity: Identity;
  faces: { face: Face; asset_path: string; crop_path: string }[];
}

export interface ImportReport {
  added: number;
  already_present: number;
  failed: number;
  assets: Asset[];
}

export const api = {
  dbStatus: () =>
    invoke<{ ok: boolean; db_path: string; tables: string[] }>("db_status"),
  listProjects: () => invoke<Project[]>("list_projects"),
  createProject: (name: string) =>
    invoke<Project>("create_project", { name }),
  importFolder: (projectId: number, folder: string) =>
    invoke<ImportReport>("import_folder", { projectId, folder }),
  listAssets: (projectId: number) =>
    invoke<Asset[]>("list_assets", { projectId }),
  listFaces: (assetId: number) => invoke<Face[]>("list_faces", { assetId }),
  detectFaces: (projectId: number) =>
    invoke<{ images_processed: number; faces_found: number; failed: number }>(
      "detect_faces",
      { projectId },
    ),
  generateEmbeddings: (projectId: number) =>
    invoke<{ embedded: number; failed: number }>("generate_embeddings", {
      projectId,
    }),
  clusterIdentities: (projectId: number) =>
    invoke<{ identities: number; clustered_faces: number; single_faces: number }>(
      "cluster_identities",
      { projectId },
    ),
  listIdentities: (projectId: number) =>
    invoke<Identity[]>("list_identities", { projectId }),
  identityDetail: (identityId: number) =>
    invoke<IdentityDetail>("identity_detail", { identityId }),
  renameIdentity: (identityId: number, name: string) =>
    invoke<void>("rename_identity", { identityId, name }),
};

export async function pickFolder(): Promise<string | null> {
  const picked = await open({ directory: true, multiple: false });
  return typeof picked === "string" ? picked : null;
}

export function fileUrl(path: string | null): string {
  return path ? convertFileSrc(path) : "";
}

export function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}
