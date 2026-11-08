//! Space view E2E: after the real pipeline runs on the sample photos,
//! the space worker must place the two Einstein faces close together
//! and far from Curie, with identities attached.
use dahlia_core::jobs::WorkerConfig;
use dahlia_core::storage::Db;
use studio_lib::{pipeline, space};

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn identity_space_on_samples() {
    let root = repo_root();
    let python = root.join(".venv/bin/python");
    if !python.exists() || !root.join("workers/face/models/det_500m.onnx").exists() {
        eprintln!("skipping: venv or models not present");
        return;
    }
    let cfg = WorkerConfig {
        python: python.to_string_lossy().into_owned(),
        workers_dir: root.join("workers"),
    };
    let samples = root.join("tests/samples");
    let db = Db::open_in_memory().unwrap();
    let project = db.create_project("SpaceE2E").unwrap();
    let data_dir = std::env::temp_dir().join("dahlia-space-e2e");
    let _ = std::fs::remove_dir_all(&data_dir);
    std::fs::create_dir_all(data_dir.join("faces")).unwrap();
    for name in ["einstein_a.jpg", "einstein_b.jpg", "curie_a.jpg"] {
        db.add_asset(project.id, samples.join(name).to_str().unwrap(), None).unwrap();
    }
    pipeline::detect_faces(&db, &cfg, &data_dir.join("faces"), project.id).unwrap();
    pipeline::generate_embeddings(&db, &cfg, project.id).unwrap();
    pipeline::cluster_identities(&db, &cfg, project.id).unwrap();

    let data = space::identity_space(&db, &cfg, &data_dir.join("faces"), project.id).unwrap();
    assert_eq!(data.points.len(), 3);
    assert_eq!(data.identities.len(), 2);
    assert!(data.points.iter().all(|p| p.identity_id.is_some()));
    assert!(data.points.iter().all(|p| std::path::Path::new(&p.crop_path).exists()));

    // group positions by identity: the pair must be tighter than the gap
    let pair = data.points.iter().filter(|p| p.identity_id == data.points[0].identity_id).collect::<Vec<_>>();
    let (a, b) = if pair.len() == 2 {
        (data.points.iter().find(|p| p.identity_id != data.points[0].identity_id).unwrap(),
         pair[0])
    } else {
        (pair[0], data.points.iter().find(|p| p.identity_id != data.points[0].identity_id).unwrap())
    };
    let _ = (a, b);
    let by_id: std::collections::HashMap<i64, Vec<[f64; 3]>> = {
        let mut m: std::collections::HashMap<i64, Vec<[f64; 3]>> = Default::default();
        for p in &data.points {
            m.entry(p.identity_id.unwrap()).or_default().push(p.pos);
        }
        m
    };
    let groups: Vec<&Vec<[f64; 3]>> = by_id.values().collect();
    let two = groups.iter().find(|g| g.len() == 2).expect("a pair identity");
    let one = groups.iter().find(|g| g.len() == 1).expect("a single identity");
    assert!(dist(two[0], two[1]) < dist(two[0], one[0]));

    // Depth: with the dev model present it succeeds; otherwise the
    // graceful "model not installed" error is the contract.
    let dev_model = root.join("../dahlia-3d/models/depth-anything-v2-small.onnx");
    let face_id = data.points[0].face_id;
    if dev_model.exists() {
        unsafe { std::env::set_var("DAHLIA_DEPTH_MODEL", &dev_model) };
    let png = space::face_depth(&cfg, &data_dir, face_id).unwrap();
        assert!(std::path::Path::new(&png).exists());
    } else {
    let err = space::face_depth(&cfg, &data_dir, face_id).unwrap_err();
        assert!(err.contains("深度モデル"), "unexpected error: {err}");
    }
}
