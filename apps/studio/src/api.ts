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
  detected: boolean;
}

export interface ProcessingFailure {
  id: number;
  project_id: number;
  asset_id: number | null;
  stage: "import" | "detect" | "embed" | "crop";
  target: string;
  message: string;
  created_at: string;
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
    invoke<{
      ok: boolean;
      db_path: string;
      tables: string[];
      notice: string | null;
      app_version: string;
      log_path: string | null;
    }>("db_status"),
  listProjects: () => invoke<Project[]>("list_projects"),
  createProject: (name: string) =>
    invoke<Project>("create_project", { name }),
  deleteProject: (projectId: number) =>
    invoke<void>("delete_project", { projectId }),
  removeAsset: (assetId: number) => invoke<void>("remove_asset", { assetId }),
  importFolder: (projectId: number, folder: string) =>
    invoke<ImportReport>("import_folder", { projectId, folder }),
  listAssets: (projectId: number) =>
    invoke<Asset[]>("list_assets", { projectId }),
  listFaces: (assetId: number) => invoke<Face[]>("list_faces", { assetId }),
  detectFaces: (projectId: number) =>
    invoke<{ images_processed: number; faces_found: number; failed: number; skipped: number }>(
      "detect_faces",
      { projectId },
    ),
  redetectFaces: (projectId: number) =>
    invoke<{ images_processed: number; faces_found: number; failed: number; skipped: number }>(
      "redetect_faces",
      { projectId },
    ),
  generateEmbeddings: (projectId: number) =>
    invoke<{ embedded: number; failed: number; skipped: number }>("generate_embeddings", {
      projectId,
    }),
  listFailures: (projectId: number) =>
    invoke<ProcessingFailure[]>("list_failures", { projectId }),
  clearFailures: (projectId: number) =>
    invoke<void>("clear_failures", { projectId }),
  clusterIdentities: (projectId: number) =>
    invoke<{
      identities: number;
      clustered_faces: number;
      single_faces: number;
      names_carried_over: number;
    }>("cluster_identities", { projectId }),
  listIdentities: (projectId: number) =>
    invoke<Identity[]>("list_identities", { projectId }),
  identityDetail: (identityId: number) =>
    invoke<IdentityDetail>("identity_detail", { identityId }),
  renameIdentity: (identityId: number, name: string) =>
    invoke<void>("rename_identity", { identityId, name }),
  mergeIdentities: (targetId: number, sourceId: number) =>
    invoke<void>("merge_identities", { targetId, sourceId }),
  moveFaceToIdentity: (faceId: number, targetIdentityId: number) =>
    invoke<void>("move_face_to_identity", { faceId, targetIdentityId }),
  identitySpace: (projectId: number) =>
    invoke<SpaceData>("identity_space", { projectId }),
  faceDepth: (faceId: number) => invoke<string>("face_depth", { faceId }),
  identityFaceMesh: (identityId: number) =>
    invoke<FaceMeshData>("identity_face_mesh", { identityId }),
};

export interface SpacePoint {
  face_id: number;
  identity_id: number | null;
  pos: [number, number, number];
  confidence: number;
  asset_path: string;
  crop_path: string;
}

export interface SpaceData {
  points: SpacePoint[];
  identities: Identity[];
  method: string;
}

export interface FaceMeshData {
  face_id: number;
  crop_path: string;
  points: [number, number, number][];
  uvs: [number, number][];
  triangles: [number, number, number][];
  blendshapes: [string, number][];
}

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
