/* 3D face preview for one identity: a photo-textured face mesh built
   from MediaPipe FaceLandmarker landmarks (478 points) computed by
   the facemesh worker. Lazy-loaded together with three.js. */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { api, fileUrl, type FaceMeshData } from "./api";
import "./FaceMesh3D.css";

export default function FaceMesh3D({ identityId }: { identityId: number }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [mesh, setMesh] = useState<FaceMeshData | null>(null);
  const [error, setError] = useState("");
  const [wireframe, setWireframe] = useState(false);
  const wireRef = useRef<THREE.MeshStandardMaterial | null>(null);

  useEffect(() => {
    api
      .identityFaceMesh(identityId)
      .then(setMesh)
      .catch((e) => setError(String(e)));
  }, [identityId]);

  useEffect(() => {
    if (!mesh || !mountRef.current) return;
    const mount = mountRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14151c);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
    camera.position.set(0.6, 0.25, 3.4);
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.autoRotate = true;
    orbit.autoRotateSpeed = 1.2;

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(2, 2.5, 3);
    scene.add(key);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const rim = new THREE.DirectionalLight(0x88aaff, 0.8);
    rim.position.set(-3, 1, -2);
    scene.add(rim);

    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(mesh.points.flat());
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    if (mesh.uvs.length === mesh.points.length) {
      geo.setAttribute("uv", new THREE.BufferAttribute(new Float32Array(mesh.uvs.flat()), 2));
    }
    geo.setIndex(mesh.triangles.flat());
    geo.computeVertexNormals();

    const tex = new THREE.TextureLoader().load(fileUrl(mesh.crop_path));
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.MeshStandardMaterial({
      map: tex,
      roughness: 0.75,
      metalness: 0,
      side: THREE.DoubleSide,
      flatShading: false,
    });
    wireRef.current = mat;
    const faceMesh = new THREE.Mesh(geo, mat);
    scene.add(faceMesh);

    const ptsGeo = new THREE.BufferGeometry();
    ptsGeo.setAttribute("position", new THREE.BufferAttribute(pos.slice(), 3));
    const points = new THREE.Points(
      ptsGeo,
      new THREE.PointsMaterial({ color: 0xffffff, size: 0.012, transparent: true, opacity: 0.55 }),
    );
    scene.add(points);

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
      orbit.update();
      renderer.render(scene, camera);
    });
    return () => {
      ro.disconnect();
      renderer.setAnimationLoop(null);
      orbit.dispose();
      renderer.dispose();
      geo.dispose();
      ptsGeo.dispose();
      tex.dispose();
      mount.removeChild(renderer.domElement);
      wireRef.current = null;
    };
  }, [mesh]);

  useEffect(() => {
    if (wireRef.current) wireRef.current.wireframe = wireframe;
  }, [wireframe, mesh]);

  if (error) return <div className="error">{error}</div>;
  if (!mesh) return <p className="hint">3D顔データを計算中…（初回は少し時間がかかります）</p>;

  const topShapes = mesh.blendshapes.filter(([, s]) => s > 0.03).slice(0, 6);
  return (
    <div className="fm-wrap">
      <div className="fm-stage">
        <div className="fm-mount" ref={mountRef} />
        <label className="fm-toggle">
          <input
            type="checkbox"
            checked={wireframe}
            onChange={(e) => setWireframe(e.target.checked)}
          />
          ワイヤーフレーム
        </label>
      </div>
      <div className="fm-side">
        <h4>表情パラメータ（Blendshapes）</h4>
        {topShapes.length === 0 ? (
          <p className="hint">目立つ表情の偏りはありません（中立に近い顔です）。</p>
        ) : (
          topShapes.map(([name, score]) => (
            <div key={name} className="fm-bar-row">
              <span className="fm-bar-name">{name}</span>
              <div className="fm-bar">
                <div style={{ width: `${Math.round(score * 100)}%` }} />
              </div>
              <span className="fm-bar-val">{Math.round(score * 100)}%</span>
            </div>
          ))
        )}
        <p className="hint">
          MediaPipe FaceLandmarker が推定した478点の顔ランドマークから組み立てた3Dメッシュです。写真の質感をそのまま貼っています。ドラッグで回転できます。
        </p>
      </div>
    </div>
  );
}
