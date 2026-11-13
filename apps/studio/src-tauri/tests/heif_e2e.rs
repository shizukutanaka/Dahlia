//! End-to-end HEIC test (W87): a HEIC photo flows through import
//! (proxy conversion + thumbnail), detection and face cropping.
//!
//! Requires the worker venv with pillow-heif installed and the face
//! model; skipped gracefully when any of them is absent.
use dahlia_core::jobs::{JobHooks, WorkerConfig};
use dahlia_core::storage::Db;
use studio_lib::{import, pipeline};

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

#[test]
fn heic_photo_imports_detects_and_crops() {
    let root = repo_root();
    let python = root.join(".venv/bin/python");
    let face_model = root.join("workers/face/models/det_500m.onnx");
    if !python.exists() || !face_model.exists() {
        eprintln!("skipping: venv or models not present");
        return;
    }
    let cfg = WorkerConfig {
        python: python.to_string_lossy().into_owned(),
        workers_dir: root.join("workers"),
    };
    // pillow-heif must be importable by the venv python.
    let probe = std::process::Command::new(&cfg.python)
        .args(["-c", "import pillow_heif"])
        .output();
    if !matches!(probe, Ok(ref o) if o.status.success()) {
        eprintln!("skipping: pillow-heif not installed in the venv");
        return;
    }

    // Turn the Einstein sample into a HEIC photo (as an iPhone
    // would deliver it) inside a fresh import folder.
    let base = std::env::temp_dir().join("dahlia-heif-e2e");
    let _ = std::fs::remove_dir_all(&base);
    let import_dir = base.join("photos");
    std::fs::create_dir_all(&import_dir).unwrap();
    let heic_path = import_dir.join("einstein.heic");
    let jpg_path = root.join("tests/samples/einstein_a.jpg");
    let script = format!(
        "import pillow_heif; pillow_heif.register_heif_opener(); \
         from PIL import Image; Image.open({jpg:?}).save({heic:?}, quality=90)",
        jpg = jpg_path.to_string_lossy(),
        heic = heic_path.to_string_lossy(),
    );
    let gen_out = std::process::Command::new(&cfg.python)
        .args(["-c", &script])
        .output()
        .unwrap();
    assert!(
        gen_out.status.success(),
        "HEIC generation failed: {gen_out:?}"
    );
    assert!(heic_path.exists());

    let db = Db::open_in_memory().unwrap();
    let project = db.create_project("HEIC E2E").unwrap();
    let thumbs = base.join("thumbnails");
    let proxies = base.join("proxies");
    let crops = base.join("faces");

    // Import: the asset registers, a JPEG proxy is converted and a
    // thumbnail is generated from it.
    let report = import::import_folder_hooked(
        &db,
        &cfg,
        &thumbs,
        &proxies,
        project.id,
        &import_dir,
        &JobHooks::none(),
    )
    .unwrap();
    assert_eq!(report.added, 1);
    assert_eq!(report.failed, 0);
    let assets = db.list_assets(project.id).unwrap();
    assert_eq!(assets.len(), 1);
    let asset = &assets[0];
    let thumbnail = asset.thumbnail.clone().expect("thumbnail must be set");
    assert!(
        std::path::Path::new(&thumbnail).exists(),
        "thumbnail file must exist"
    );
    assert!(
        proxies.join(format!("{}.jpg", asset.id)).exists(),
        "JPEG proxy must exist"
    );

    // Detection runs on the original HEIC (workers decode it), and
    // the face crop is cut from the proxy.
    let report = pipeline::detect_faces(&db, &cfg, &crops, &proxies, project.id).unwrap();
    assert_eq!(report.failed, 0);
    assert_eq!(report.faces_found, 1);
    let faces = db.list_faces(asset.id).unwrap();
    assert_eq!(faces.len(), 1);
    assert!(
        crops.join(format!("{}.jpg", faces[0].id)).exists(),
        "face crop must be cut from the proxy"
    );
}
