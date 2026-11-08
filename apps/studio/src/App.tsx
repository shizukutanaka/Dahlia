import { Suspense, lazy, useCallback, useEffect, useState } from "react";
import {
  api,
  baseName,
  fileUrl,
  pickFolder,
  type Asset,
  type Identity,
  type IdentityDetail,
  type Project,
} from "./api";
const Space3D = lazy(() => import("./Space3D"));
import "./App.css";

type View =
  | { kind: "projects" }
  | { kind: "project"; project: Project }
  | { kind: "identity"; project: Project; identityId: number }
  | { kind: "space"; project: Project };

function App() {
  const [view, setView] = useState<View>({ kind: "projects" });
  const [dbPath, setDbPath] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    api
      .dbStatus()
      .then((s) => setDbPath(s.db_path))
      .catch((e) => setError(`データベースに接続できません: ${e}`));
  }, []);

  return (
    <div className="app">
      <header>
        <h1 onClick={() => setView({ kind: "projects" })}>🌸 Dahlia</h1>
        <span className="subtitle">人物 Identity 管理</span>
      </header>
      {error && <div className="error">{error}</div>}
      {view.kind === "projects" && (
        <ProjectList onOpen={(p) => setView({ kind: "project", project: p })} />
      )}
      {view.kind === "project" && (
        <ProjectView
          project={view.project}
          onBack={() => setView({ kind: "projects" })}
          onOpenIdentity={(id) =>
            setView({ kind: "identity", project: view.project, identityId: id })
          }
          onOpenSpace={() => setView({ kind: "space", project: view.project })}
        />
      )}
      {view.kind === "space" && (
        <Suspense fallback={<p className="hint">3D空間を読み込み中…</p>}>
          <Space3D
            project={view.project}
            onBack={() => setView({ kind: "project", project: view.project })}
          />
        </Suspense>
      )}
      {view.kind === "identity" && (
        <IdentityView
          identityId={view.identityId}
          onBack={() => setView({ kind: "project", project: view.project })}
        />
      )}
      <footer>データの保存先: {dbPath}</footer>
    </div>
  );
}

function ProjectList({ onOpen }: { onOpen: (p: Project) => void }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(() => {
    api.listProjects().then(setProjects).catch((e) => setError(String(e)));
  }, []);
  useEffect(load, [load]);

  async function create() {
    if (!name.trim()) return;
    try {
      await api.createProject(name.trim());
      setName("");
      load();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <main>
      <h2>プロジェクト</h2>
      {error && <div className="error">{error}</div>}
      <div className="row">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="新しいプロジェクト名（例: 推しキャラA）"
          onKeyDown={(e) => e.key === "Enter" && create()}
        />
        <button onClick={create} disabled={!name.trim()}>
          作成する
        </button>
      </div>
      {projects.length === 0 && (
        <p className="hint">
          まだプロジェクトがありません。名前を付けて作成するところから始めましょう。
        </p>
      )}
      <div className="cards">
        {projects.map((p) => (
          <button key={p.id} className="card" onClick={() => onOpen(p)}>
            <strong>{p.name}</strong>
            <span className="hint">{p.created_at}</span>
          </button>
        ))}
      </div>
    </main>
  );
}

function ProjectView({
  project,
  onBack,
  onOpenIdentity,
  onOpenSpace,
}: {
  project: Project;
  onBack: () => void;
  onOpenIdentity: (id: number) => void;
  onOpenSpace: () => void;
}) {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [faceCounts, setFaceCounts] = useState<Record<number, number>>({});
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const loadAssets = useCallback(async () => {
    const list = await api.listAssets(project.id);
    setAssets(list);
    const counts: Record<number, number> = {};
    await Promise.all(
      list.map(async (a) => {
        const faces = await api.listFaces(a.id);
        counts[a.id] = faces.length;
      }),
    );
    setFaceCounts(counts);
  }, [project.id]);

  const loadIdentities = useCallback(() => {
    api.listIdentities(project.id).then(setIdentities).catch(() => {});
  }, [project.id]);

  useEffect(() => {
    loadAssets().catch((e) => setError(String(e)));
    loadIdentities();
  }, [loadAssets, loadIdentities]);

  async function run<T>(label: string, job: () => Promise<T>, done: (r: T) => string) {
    setBusy(label);
    setError("");
    setStatus("");
    try {
      const r = await job();
      setStatus(done(r));
    } catch (e) {
      setError(`${label}に失敗しました: ${e}`);
    } finally {
      setBusy("");
    }
  }

  async function importFolder() {
    const folder = await pickFolder();
    if (!folder) return;
    await run("写真の読み込み", () => api.importFolder(project.id, folder), (r) => {
      loadAssets();
      return `読み込み完了: 新しく ${r.added} 枚を追加（登録済み ${r.already_present} 枚、失敗 ${r.failed} 枚）`;
    });
  }

  return (
    <main>
      <button className="link" onClick={onBack}>
        ← プロジェクト一覧へ
      </button>
      <h2>{project.name}</h2>
      {error && <div className="error">{error}</div>}
      {status && <div className="ok">{status}</div>}

      <section>
        <h3>写真（{assets.length}枚）</h3>
        <div className="row">
          <button onClick={importFolder} disabled={!!busy}>
            📁 写真フォルダを追加
          </button>
          <button
            onClick={() =>
              run("顔の検出", () => api.detectFaces(project.id), (r) => {
                loadAssets();
                return `検出完了: ${r.images_processed} 枚から顔が ${r.faces_found} 個見つかりました（失敗 ${r.failed} 枚）`;
              })
            }
            disabled={!!busy || assets.length === 0}
          >
            🔍 顔を検出する
          </button>
          <button
            onClick={() =>
              run("特徴量の取得", () => api.generateEmbeddings(project.id), (r) =>
                `特徴量を取得しました: ${r.embedded} 件（失敗 ${r.failed} 件）`,
              )
            }
            disabled={!!busy || assets.length === 0}
          >
            ✨ 特徴量を取得する
          </button>
          <button
            onClick={() =>
              run("人物の分類", () => api.clusterIdentities(project.id), (r) => {
                loadIdentities();
                return `分類完了: ${r.identities} 人の人物が見つかりました`;
              })
            }
            disabled={!!busy || assets.length === 0}
          >
            👥 人物を分類する
          </button>
          {busy && <span className="hint">{busy}中…しばらくお待ちください</span>}
        </div>
        {assets.length === 0 ? (
          <p className="hint">まだ写真がありません。「写真フォルダを追加」から読み込んでください。</p>
        ) : (
          <div className="grid">
            {assets.map((a) => (
              <figure key={a.id} className="thumb">
                <img src={fileUrl(a.thumbnail ?? a.path)} alt={baseName(a.path)} loading="lazy" />
                <figcaption>
                  {baseName(a.path)}
                  {faceCounts[a.id] ? <span className="badge">顔 {faceCounts[a.id]}</span> : null}
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </section>

      <section>
        <h3>
          人物 Identity（{identities.length}人）
          {identities.length > 0 && (
            <button className="s3-open" onClick={onOpenSpace} style={{ marginLeft: 12 }}>
              🧊 3D空間で見る
            </button>
          )}
        </h3>
        {identities.length === 0 ? (
          <p className="hint">
            まだ人物が登録されていません。写真を読み込み → 顔を検出 → 特徴量を取得 → 人物を分類、の順に進めてください。
          </p>
        ) : (
          <div className="cards">
            {identities.map((i) => (
              <button key={i.id} className="card" onClick={() => onOpenIdentity(i.id)}>
                <strong>{i.name}</strong>
                <span className="hint">
                  顔 {i.face_count} 件 / 品質スコア {i.quality_score.toFixed(2)}
                </span>
              </button>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function IdentityView({ identityId, onBack }: { identityId: number; onBack: () => void }) {
  const [detail, setDetail] = useState<IdentityDetail | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(() => {
    api
      .identityDetail(identityId)
      .then((d) => {
        setDetail(d);
        setName(d.identity.name);
      })
      .catch((e) => setError(String(e)));
  }, [identityId]);
  useEffect(load, [load]);

  async function rename() {
    if (!name.trim()) return;
    try {
      await api.renameIdentity(identityId, name.trim());
      load();
    } catch (e) {
      setError(String(e));
    }
  }

  if (!detail) return <main>{error ? <div className="error">{error}</div> : <p>読み込み中…</p>}</main>;
  return (
    <main>
      <button className="link" onClick={onBack}>
        ← プロジェクトへ戻る
      </button>
      <h2>{detail.identity.name}</h2>
      {error && <div className="error">{error}</div>}
      <div className="row">
        <input value={name} onChange={(e) => setName(e.target.value)} />
        <button onClick={rename} disabled={!name.trim() || name === detail.identity.name}>
          名前を変更
        </button>
        <span className="hint">
          顔 {detail.identity.face_count} 件 / 品質スコア {detail.identity.quality_score.toFixed(2)}
        </span>
      </div>
      <div className="grid">
        {detail.faces.map(({ face, crop_path, asset_path }) => (
          <figure key={face.id} className="thumb">
            <img src={fileUrl(crop_path)} alt={baseName(asset_path)} loading="lazy" />
            <figcaption>{baseName(asset_path)}</figcaption>
          </figure>
        ))}
      </div>
    </main>
  );
}

export default App;
