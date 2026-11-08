/* 3D Identity Space, built into Dahlia.
   Same visualisation as the standalone Dahlia 3D explorer: face
   embeddings (512-d) reduced to 3D, one cluster per identity.
   Positions come from the `identity_space` command (space worker);
   depth maps for the relief view from `face_depth`. */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { ConvexGeometry } from "three/addons/geometries/ConvexGeometry.js";
import {
  api,
  baseName,
  fileUrl,
  type Identity,
  type Project,
  type SpaceData,
  type SpacePoint,
} from "./api";
import "./Space3D.css";

const PALETTE = [
  "#f472b6", "#38bdf8", "#a3e635", "#fbbf24", "#c084fc",
  "#34d399", "#fb7185", "#60a5fa", "#facc15", "#4ade80",
];

interface ThreeApi {
  focus: (id: number | null) => void;
  setPhotos: (on: boolean) => void;
  setHulls: (on: boolean) => void;
  setRotate: (on: boolean) => void;
  enterRelief: (p: SpacePoint, depthUrl: string) => void;
  exitRelief: () => void;
}

export default function Space3D({
  project,
  onBack,
}: {
  project: Project;
  onBack: () => void;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const threeRef = useRef<ThreeApi | null>(null);
  const [data, setData] = useState<SpaceData | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<SpacePoint | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const [showPhotos, setShowPhotos] = useState(true);
  const [showHulls, setShowHulls] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);
  const [reliefPoint, setReliefPoint] = useState<SpacePoint | null>(null);
  const [depthBusy, setDepthBusy] = useState(false);
  const [depthError, setDepthError] = useState("");

  useEffect(() => {
    api
      .identitySpace(project.id)
      .then(setData)
      .catch((e) => setError(`3Dデータを読み込めませんでした: ${e}`));
  }, [project.id]);

  const colorOf = (identityId: number | null): string => {
    if (!data || identityId === null) return "#9aa0ae";
    const idx = data.identities.findIndex((i) => i.id === identityId);
    return PALETTE[(idx + PALETTE.length) % PALETTE.length];
  };
  const nameOf = (identityId: number | null): string => {
    if (!data || identityId === null) return "未分類";
    return data.identities.find((i) => i.id === identityId)?.name ?? "未分類";
  };

  useEffect(() => {
    if (!data || !mountRef.current || data.points.length === 0) return;
    const mount = mountRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14151c);
    scene.fog = new THREE.Fog(0x14151c, 16, 30);
    const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 100);
    const HOME_POS = new THREE.Vector3(0.5, 2.4, 11);
    const HOME_TGT = new THREE.Vector3(0, 0, 0);
    camera.position.copy(HOME_POS);
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.autoRotate = true;
    orbit.autoRotateSpeed = 0.7;

    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.5;
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(4, 6, 5);
    scene.add(key);
    scene.add(new THREE.AmbientLight(0x8899bb, 0.25));

    const world = new THREE.Group();
    scene.add(world);
    const reliefGroup = new THREE.Group();
    reliefGroup.visible = false;
    scene.add(reliefGroup);

    const colorFor = (iid: number | null) =>
      iid === null
        ? "#9aa0ae"
        : PALETTE[
            (data.identities.findIndex((i) => i.id === iid) + PALETTE.length) %
              PALETTE.length
          ];

    const textSprite = (text: string, color: string) => {
      const c = document.createElement("canvas");
      c.width = 512; c.height = 112;
      const g = c.getContext("2d")!;
      g.font = "700 52px 'Segoe UI', sans-serif";
      g.textAlign = "center"; g.textBaseline = "middle";
      g.shadowColor = "rgba(0,0,0,.9)"; g.shadowBlur = 10;
      g.fillStyle = color;
      g.fillText(text, 256, 56);
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace;
      const sp = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }),
      );
      sp.scale.set(2.3, 0.5, 1);
      return sp;
    };

    const circularPhoto = (url: string, ring: string) => {
      const c = document.createElement("canvas");
      c.width = c.height = 256;
      const g = c.getContext("2d")!;
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace;
      const img = new Image();
      img.onload = () => {
        g.save();
        g.beginPath(); g.arc(128, 128, 118, 0, Math.PI * 2); g.clip();
        g.drawImage(img, 0, 0, 256, 256);
        g.restore();
        g.beginPath(); g.arc(128, 128, 121, 0, Math.PI * 2);
        g.lineWidth = 9; g.strokeStyle = ring; g.stroke();
        tex.needsUpdate = true;
      };
      img.src = url;
      return tex;
    };

    interface Entry {
      spheres: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>[];
      sprites: THREE.Sprite[];
      lines?: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
      hull?: THREE.Mesh;
      centroid: THREE.Vector3;
    }
    const entries = new Map<number, Entry>();
    const pickables: THREE.Object3D[] = [];
    const sphereGeo = new THREE.SphereGeometry(1, 24, 18);

    for (const p of data.points) {
      const key2 = p.identity_id ?? -1;
      const entry = entries.get(key2) ?? {
        spheres: [],
        sprites: [],
        centroid: new THREE.Vector3(),
      };
      entries.set(key2, entry);
      const color = colorFor(p.identity_id);
      const mat = new THREE.MeshStandardMaterial({
        color, roughness: 0.3, metalness: 0.1,
        emissive: color, emissiveIntensity: 0.25, transparent: true,
      });
      const sphere = new THREE.Mesh(sphereGeo, mat);
      sphere.position.set(...p.pos);
      sphere.scale.setScalar(0.07 + p.confidence * 0.06);
      sphere.userData.point = p;
      world.add(sphere);
      entry.spheres.push(sphere);
      pickables.push(sphere);

      const sp = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: circularPhoto(fileUrl(p.crop_path), color),
          transparent: true, depthWrite: false,
        }),
      );
      sp.position.set(p.pos[0], p.pos[1] + 0.34, p.pos[2]);
      sp.scale.setScalar(0.52);
      sp.userData.point = p;
      world.add(sp);
      entry.sprites.push(sp);
      pickables.push(sp);
    }

    for (const [iid, entry] of entries) {
      const pts = data.points.filter((p) => (p.identity_id ?? -1) === iid);
      const centroid = new THREE.Vector3();
      pts.forEach((p) => centroid.add(new THREE.Vector3(...p.pos)));
      centroid.divideScalar(pts.length);
      entry.centroid = centroid;
      const color = colorFor(iid === -1 ? null : iid);
      const linePos: number[] = [];
      for (const p of pts) linePos.push(...p.pos, centroid.x, centroid.y, centroid.z);
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePos, 3));
      const lines = new THREE.LineSegments(
        lineGeo,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 }),
      );
      world.add(lines);
      entry.lines = lines;
      const core = new THREE.Mesh(
        new THREE.SphereGeometry(0.05, 16, 12),
        new THREE.MeshBasicMaterial({ color }),
      );
      core.position.copy(centroid);
      world.add(core);
      const info = data.identities.find((i) => i.id === iid);
      const label = textSprite(`${info?.name ?? "未分類"}（${pts.length}）`, color);
      label.position.copy(centroid);
      label.position.y += 0.7;
      world.add(label);
      if (pts.length >= 3) {
        try {
          entry.hull = new THREE.Mesh(
            new ConvexGeometry(pts.map((p) => new THREE.Vector3(...p.pos))),
            new THREE.MeshPhysicalMaterial({
              color, transparent: true, opacity: 0.07, roughness: 0.4,
              metalness: 0, depthWrite: false, side: THREE.DoubleSide,
            }),
          );
          world.add(entry.hull);
        } catch { /* collinear: no hull */ }
      }
    }

    let camGoal: THREE.Vector3 | null = null;
    const reliefMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(3.1, 3.1, 190, 190),
      new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0 }),
    );
    reliefMesh.rotation.x = -0.12;
    reliefGroup.add(reliefMesh);
    const rLight = new THREE.DirectionalLight(0xffffff, 2.0);
    rLight.position.set(2.5, 3, 4);
    reliefGroup.add(rLight);
    reliefGroup.add(new THREE.AmbientLight(0xffffff, 0.35));
    const texLoader = new THREE.TextureLoader();

    threeRef.current = {
      focus: (id) => {
        for (const [iid, entry] of entries) {
          const on = id === null || iid === id;
          entry.spheres.forEach((s) => { s.material.opacity = on ? 1 : 0.13; });
          entry.sprites.forEach((s) => { s.material.opacity = on ? 1 : 0.1; });
          if (entry.lines) entry.lines.material.opacity = on ? 0.4 : 0.05;
        }
        if (id !== null && entries.get(id)) camGoal = entries.get(id)!.centroid.clone();
      },
      setPhotos: (on) => {
        for (const e of entries.values()) e.sprites.forEach((s) => { s.visible = on; });
      },
      setHulls: (on) => {
        for (const e of entries.values()) if (e.hull) e.hull.visible = on;
      },
      setRotate: (on) => { orbit.autoRotate = on && !reliefGroup.visible; },
      enterRelief: (p, depthUrl) => {
        const map = texLoader.load(fileUrl(p.crop_path));
        map.colorSpace = THREE.SRGBColorSpace;
        reliefMesh.material.map = map;
        reliefMesh.material.displacementMap = texLoader.load(depthUrl);
        reliefMesh.material.displacementScale = 0.7;
        reliefMesh.material.needsUpdate = true;
        world.visible = false;
        reliefGroup.visible = true;
        orbit.autoRotate = false;
        camera.position.set(2.0, 0.9, 4.1);
        orbit.target.set(0, 0, 0);
      },
      exitRelief: () => {
        world.visible = true;
        reliefGroup.visible = false;
        camera.position.copy(HOME_POS);
        orbit.target.copy(HOME_TGT);
      },
    };

    const raycaster = new THREE.Raycaster();
    let downPos: [number, number] | null = null;
    const dom = renderer.domElement;
    const onDown = (e: PointerEvent) => { downPos = [e.clientX, e.clientY]; };
    const onUp = (e: PointerEvent) => {
      if (!downPos) return;
      const moved = Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]);
      downPos = null;
      if (moved > 5 || reliefGroup.visible) return;
      const rect = dom.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObjects(pickables, false);
      if (hits[0]) setSelected(hits[0].object.userData.point as SpacePoint);
    };
    dom.addEventListener("pointerdown", onDown);
    dom.addEventListener("pointerup", onUp);

    const resize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    renderer.setAnimationLoop(() => {
      if (camGoal) {
        orbit.target.lerp(camGoal, 0.06);
        if (orbit.target.distanceTo(camGoal) < 0.02) camGoal = null;
      }
      orbit.update();
      renderer.render(scene, camera);
    });

    return () => {
      ro.disconnect();
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointerup", onUp);
      renderer.setAnimationLoop(null);
      orbit.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      threeRef.current = null;
    };
  }, [data]);

  const toggleFocus = (id: number | null) => {
    const next = focused === id ? null : id;
    setFocused(next);
    threeRef.current?.focus(next);
  };

  const openRelief = async (p: SpacePoint) => {
    setDepthBusy(true);
    setDepthError("");
    try {
      const png = await api.faceDepth(p.face_id);
      setReliefPoint(p);
      threeRef.current?.enterRelief(p, fileUrl(png));
    } catch (e) {
      setDepthError(String(e));
    } finally {
      setDepthBusy(false);
    }
  };

  const closeRelief = () => {
    setReliefPoint(null);
    threeRef.current?.exitRelief();
    threeRef.current?.setRotate(autoRotate);
  };

  const counts = new Map<number, number>();
  data?.points.forEach((p) => {
    if (p.identity_id !== null) counts.set(p.identity_id, (counts.get(p.identity_id) ?? 0) + 1);
  });

  return (
    <main>
      <button className="link" onClick={onBack}>← プロジェクトに戻る</button>
      <h2>🧊 3D空間 - {project.name}</h2>
      {error && <div className="error">{error}</div>}
      {data && data.points.length === 0 && (
        <p className="hint">
          まだ特徴量のある顔がありません。先に「顔を検出する → 特徴量を取得する → 人物を分類する」を実行してください。
        </p>
      )}
      {data && data.points.length > 0 && (
        <>
          <div className="row s3-tools">
            <button
              className={autoRotate ? "s3-on" : ""}
              onClick={() => {
                const next = !autoRotate;
                setAutoRotate(next);
                threeRef.current?.setRotate(next);
              }}
            >
              自動回転
            </button>
            <button
              className={showPhotos ? "s3-on" : ""}
              onClick={() => {
                const next = !showPhotos;
                setShowPhotos(next);
                threeRef.current?.setPhotos(next);
              }}
            >
              写真を表示
            </button>
            <button
              className={showHulls ? "s3-on" : ""}
              onClick={() => {
                const next = !showHulls;
                setShowHulls(next);
                threeRef.current?.setHulls(next);
              }}
            >
              クラスタの殻
            </button>
            <span className="hint">
              顔 {data.points.length} 件 / 人物 {data.identities.length} 人（{data.method === "umap" ? "UMAP" : "PCA"}で3次元化）
            </span>
          </div>
          <div className="s3-layout">
            <aside className="s3-side">
              <h3>人物 Identity</h3>
              <ul className="s3-list">
                {data.identities.map((i: Identity) => (
                  <li
                    key={i.id}
                    className={focused === i.id ? "s3-focused" : ""}
                    onClick={() => toggleFocus(i.id)}
                  >
                    <span className="s3-dot" style={{ background: colorOf(i.id) }} />
                    {i.name}
                    <span className="s3-count">{counts.get(i.id) ?? 0} 件</span>
                  </li>
                ))}
              </ul>
              <p className="hint">
                顔の特徴量を3次元に圧縮した空間です。似ている顔ほど近くに集まります。点をクリックすると詳細が見られます。
              </p>
            </aside>
            <div className="s3-stage">
              <div className="s3-mount" ref={mountRef} />
              {reliefPoint && (
                <div className="s3-reliefbar">
                  <span>{nameOf(reliefPoint.identity_id)} - {baseName(reliefPoint.asset_path)}（AI深度推定による立体化）</span>
                  <button onClick={closeRelief}>← 3D空間に戻る</button>
                </div>
              )}
            </div>
            <aside className="s3-side">
              <h3>選択中の顔</h3>
              {!selected ? (
                <p className="hint">まだ選択されていません。3D空間の点（顔）をクリックしてください。</p>
              ) : (
                <div>
                  <img className="s3-photo" src={fileUrl(selected.crop_path)} alt="" />
                  <div className="s3-name" style={{ color: colorOf(selected.identity_id) }}>
                    {nameOf(selected.identity_id)}
                  </div>
                  <div className="hint">
                    元の写真: {baseName(selected.asset_path)}<br />
                    顔の検出の確かさ: {(selected.confidence * 100).toFixed(0)}%
                  </div>
                  {depthError && <div className="error">{depthError}</div>}
                  <button
                    className="s3-relief-btn"
                    disabled={depthBusy}
                    onClick={() => openRelief(selected)}
                  >
                    {depthBusy ? "深度を計算中…" : "🧊 この顔を3Dレリーフで見る"}
                  </button>
                </div>
              )}
            </aside>
          </div>
        </>
      )}
    </main>
  );
}
