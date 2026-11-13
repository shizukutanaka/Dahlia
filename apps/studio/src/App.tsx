import { Suspense, lazy, useCallback, useEffect, useState } from "react";
import {
  api,
  baseName,
  fileUrl,
  pickFolder,
  type Asset,
  type Identity,
  type IdentityDetail,
  type JobProgress,
  type ProcessingFailure,
  type Project,
} from "./api";

const STAGE_LABEL: Record<ProcessingFailure["stage"], string> = {
  import: "写真の読み込み",
  detect: "顔の検出",
  embed: "特徴量の取得",
  crop: "顔の切り抜き",
};
const Space3D = lazy(() => import("./Space3D"));
const FaceMesh3D = lazy(() => import("./FaceMesh3D"));
import "./App.css";

type View =
  | { kind: "projects" }
  | { kind: "project"; project: Project }
  | { kind: "identity"; project: Project; identityId: number }
  | { kind: "space"; project: Project };

function App() {
  const [view, setView] = useState<View>({ kind: "projects" });
  const [dbPath, setDbPath] = useState<string>("");
  const [appVersion, setAppVersion] = useState<string>("");
  const [logPath, setLogPath] = useState<string | null>(null);
  const [error, setError] = useState<string>("");
  const [notice, setNotice] = useState<string>("");

  useEffect(() => {
    api
      .dbStatus()
      .then((s) => {
        setDbPath(s.db_path);
        setAppVersion(s.app_version);
        setLogPath(s.log_path);
        if (s.notice) setNotice(s.notice);
      })
      .catch((e) => setError(`データベースに接続できません: ${e}`));
  }, []);

  return (
    <div className="app">
      <header>
        <h1 onClick={() => setView({ kind: "projects" })}>🌸 Dahlia</h1>
        <span className="subtitle">人物 Identity 管理</span>
      </header>
      {error && <div className="error">{error}</div>}
      {notice && <div className="ok">{notice}</div>}
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
          projectId={view.project.id}
          onBack={() => setView({ kind: "project", project: view.project })}
        />
      )}
      <footer>
        Dahlia v{appVersion} ／ データの保存先: {dbPath}
        {logPath && <> ／ ログ: {logPath}</>}
      </footer>
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

  async function remove(p: Project) {
    const ok = window.confirm(
      `プロジェクト「${p.name}」を削除しますか？\n登録した写真・顔・人物のデータがすべて消えます。\n（元の写真ファイルは消えません）`,
    );
    if (!ok) return;
    try {
      await api.deleteProject(p.id);
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
          <div key={p.id} className="card-wrap">
            <button className="card" onClick={() => onOpen(p)}>
              <strong>{p.name}</strong>
              <span className="hint">{p.created_at}</span>
            </button>
            <button
              className="card-del"
              title="このプロジェクトを削除"
              onClick={() => remove(p)}
            >
              削除
            </button>
          </div>
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
  const [failures, setFailures] = useState<ProcessingFailure[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState("");
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const [error, setError] = useState("");

  // While a job runs, poll its progress (the backend reports
  // counts through the job hooks) so the user sees how far it is
  // and can stop it.
  useEffect(() => {
    if (!busy) return;
    let alive = true;
    const tick = () => {
      api
        .jobProgress()
        .then((p) => {
          if (alive) setProgress(p);
        })
        .catch(() => {});
    };
    tick();
    const t = setInterval(tick, 300);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [busy]);

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

  const loadFailures = useCallback(() => {
    api.listFailures(project.id).then(setFailures).catch(() => {});
  }, [project.id]);

  useEffect(() => {
    loadAssets().catch((e) => setError(String(e)));
    loadIdentities();
    loadFailures();
  }, [loadAssets, loadIdentities, loadFailures]);

  async function run<T>(label: string, job: () => Promise<T>, done: (r: T) => string) {
    setBusy(label);
    setError("");
    setStatus("");
    try {
      const r = await job();
      const msg = done(r);
      if (
        r &&
        typeof r === "object" &&
        "cancelled" in r &&
        (r as { cancelled: boolean }).cancelled
      ) {
        setStatus(
          "中断しました。ここまで完了した分は保存されています。もう一度実行すると、残りから進みます",
        );
      } else {
        setStatus(msg);
      }
    } catch (e) {
      setError(`${label}に失敗しました: ${e}`);
    } finally {
      setBusy("");
      setProgress(null);
      loadFailures();
    }
  }

  async function removeAsset(a: Asset) {
    const ok = window.confirm(
      `「${baseName(a.path)}」の登録を解除しますか？\n検出した顔や特徴量も消えます。\n（元の写真ファイルは消えません）`,
    );
    if (!ok) return;
    try {
      await api.removeAsset(a.id);
      loadAssets();
      loadIdentities();
      loadFailures();
    } catch (e) {
      setError(String(e));
    }
  }

  function retryStage(stage: ProcessingFailure["stage"]) {
    if (stage === "detect") {
      run("顔の検出", () => api.detectFaces(project.id), (r) => {
        loadAssets();
        return `検出完了: ${r.images_processed} 枚から顔が ${r.faces_found} 個見つかりました（失敗 ${r.failed} 枚）`;
      });
    } else if (stage === "crop") {
      // Crops are only regenerated together with detection, and the
      // normal run skips processed photos — so retrying a crop means
      // re-detecting that photo (the full re-detect covers it).
      run("顔の検出（全件）", () => api.redetectFaces(project.id), (r) => {
        loadAssets();
        return `再検出完了: ${r.images_processed} 枚から顔が ${r.faces_found} 個見つかりました（失敗 ${r.failed} 枚）`;
      });
    } else if (stage === "embed") {
      run(
        "特徴量の取得",
        () => api.generateEmbeddings(project.id),
        (r) => `特徴量を取得しました: ${r.embedded} 件（失敗 ${r.failed} 件）`,
      );
    } else {
      importFolder();
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
                const skipped =
                  r.skipped > 0 ? `（処理済みの ${r.skipped} 枚はスキップしました）` : "";
                return `検出完了: ${r.images_processed} 枚から顔が ${r.faces_found} 個見つかりました（失敗 ${r.failed} 枚）${skipped}`;
              })
            }
            disabled={!!busy || assets.length === 0}
          >
            🔍 顔を検出する
          </button>
          <button
            onClick={() => {
              const ok = window.confirm(
                "すべての写真を検出し直しますか？\n顔を作り直すため、取得済みの特徴量は消えます（続けて「特徴量を取得する」「人物を分類する」が必要になります）。",
              );
              if (!ok) return;
              run("顔の検出（全件）", () => api.redetectFaces(project.id), (r) => {
                loadAssets();
                return `再検出完了: ${r.images_processed} 枚から顔が ${r.faces_found} 個見つかりました（失敗 ${r.failed} 枚）`;
              });
            }}
            disabled={!!busy || assets.length === 0}
          >
            全件を再検出
          </button>
          <button
            onClick={() =>
              run("特徴量の取得", () => api.generateEmbeddings(project.id), (r) => {
                const skipped =
                  r.skipped > 0 ? `（取得済みの ${r.skipped} 件はスキップしました）` : "";
                return `特徴量を取得しました: ${r.embedded} 件（失敗 ${r.failed} 件）${skipped}`;
              })
            }
            disabled={!!busy || assets.length === 0}
          >
            ✨ 特徴量を取得する
          </button>
          <button
            onClick={() =>
              run("人物の分類", () => api.clusterIdentities(project.id), (r) => {
                loadIdentities();
                const carried =
                  r.names_carried_over > 0
                    ? `（${r.names_carried_over} 人は前回の名前を引き継ぎました）`
                    : "";
                return `分類完了: ${r.identities} 人の人物が見つかりました${carried}`;
              })
            }
            disabled={!!busy || assets.length === 0}
          >
            👥 人物を分類する
          </button>
          {busy && (
            <span className="job-progress">
              {progress && progress.total > 0 ? (
                <>
                  <progress max={progress.total} value={progress.done} />
                  <span className="hint">
                    {progress.phase ? `${progress.phase}：` : ""}
                    {progress.done} / {progress.total}
                  </span>
                </>
              ) : (
                <span className="hint">{busy}中…しばらくお待ちください</span>
              )}
              <button className="mini" onClick={() => api.cancelJob()}>
                中断する
              </button>
            </span>
          )}
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
                  <button
                    className="mini"
                    title="この写真の登録を解除"
                    onClick={() => removeAsset(a)}
                  >
                    登録解除
                  </button>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </section>

      {failures.length > 0 && (
        <section>
          <h3>処理に失敗した写真（{failures.length}件）</h3>
          <p className="hint">
            失敗した写真は処理から静かに消えず、ここに理由とともに残ります。原因を取り除いてから再試行してください。
          </p>
          <ul className="fail-list">
            {failures.map((f) => (
              <li key={f.id}>
                <span className="badge">{STAGE_LABEL[f.stage]}</span> {baseName(f.target)}
                <span className="hint"> — {f.message}</span>
              </li>
            ))}
          </ul>
          <div className="row">
            {[...new Set(failures.map((f) => f.stage))].map((stage) => (
              <button key={stage} onClick={() => retryStage(stage)} disabled={!!busy}>
                {STAGE_LABEL[stage]}を再試行
              </button>
            ))}
            <button
              onClick={async () => {
                await api.clearFailures(project.id);
                loadFailures();
              }}
              disabled={!!busy}
            >
              記録を消す
            </button>
          </div>
        </section>
      )}

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

function IdentityView({
  identityId,
  projectId,
  onBack,
}: {
  identityId: number;
  projectId: number;
  onBack: () => void;
}) {
  const [detail, setDetail] = useState<IdentityDetail | null>(null);
  const [showMesh, setShowMesh] = useState(false);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [others, setOthers] = useState<Identity[]>([]);
  const [mergeTarget, setMergeTarget] = useState("");
  const [moveTargets, setMoveTargets] = useState<Record<number, string>>({});

  const load = useCallback(() => {
    api
      .identityDetail(identityId)
      .then((d) => {
        setDetail(d);
        setName(d.identity.name);
      })
      .catch((e) => setError(String(e)));
    api
      .listIdentities(projectId)
      .then((list) => setOthers(list.filter((i) => i.id !== identityId)))
      .catch(() => {});
  }, [identityId, projectId]);
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

  async function merge() {
    if (!mergeTarget || !detail) return;
    const target = others.find((i) => i.id === Number(mergeTarget));
    const ok = window.confirm(
      `「${detail.identity.name}」を「${target?.name ?? ""}」に統合しますか？\nこの人物の顔はすべて統合先に移り、この人物は一覧から消えます。`,
    );
    if (!ok) return;
    try {
      await api.mergeIdentities(Number(mergeTarget), identityId);
      onBack();
    } catch (e) {
      setError(String(e));
    }
  }

  async function moveFace(faceId: number) {
    const target = moveTargets[faceId];
    if (!target) return;
    try {
      await api.moveFaceToIdentity(faceId, Number(target));
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
      {others.length > 0 && (
        <section>
          <h3>人物の整理（分類の間違いを直す）</h3>
          <div className="row">
            <span className="hint">同じ人物が分かれてしまったときは統合できます:</span>
            <select value={mergeTarget} onChange={(e) => setMergeTarget(e.target.value)}>
              <option value="">統合先の人物を選ぶ</option>
              {others.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.name}（顔 {o.face_count} 件）
                </option>
              ))}
            </select>
            <button onClick={merge} disabled={!mergeTarget}>
              この人物を統合する
            </button>
          </div>
          <p className="hint">
            1件だけ人物が違う顔は、下の一覧で顔ごとに別の人物へ移動できます。
          </p>
        </section>
      )}
      <section>
        <h3>
          3D顔プレビュー
          <button onClick={() => setShowMesh(!showMesh)} style={{ marginLeft: 12 }}>
            {showMesh ? "閉じる" : "🧑 3Dで見る"}
          </button>
        </h3>
        {showMesh && (
          <Suspense fallback={<p className="hint">3Dプレビューを読み込み中…</p>}>
            <FaceMesh3D identityId={identityId} />
          </Suspense>
        )}
      </section>
      <div className="grid">
        {detail.faces.map(({ face, crop_path, asset_path }) => (
          <figure key={face.id} className="thumb">
            <img src={fileUrl(crop_path)} alt={baseName(asset_path)} loading="lazy" />
            <figcaption>{baseName(asset_path)}</figcaption>
            {others.length > 0 && (
              <div className="face-move">
                <select
                  value={moveTargets[face.id] ?? ""}
                  onChange={(e) =>
                    setMoveTargets((m) => ({ ...m, [face.id]: e.target.value }))
                  }
                >
                  <option value="">移動先…</option>
                  {others.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.name}
                    </option>
                  ))}
                </select>
                <button
                  className="mini"
                  onClick={() => moveFace(face.id)}
                  disabled={!moveTargets[face.id]}
                >
                  移動
                </button>
              </div>
            )}
          </figure>
        ))}
      </div>
    </main>
  );
}

export default App;
