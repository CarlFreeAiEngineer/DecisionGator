//! Point the shared library source at a bundle folder and check that its model
//! and tokenizer match the manifest before they are compiled in.
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=DG_EMBED_DIR");
    let directory = PathBuf::from(std::env::var("DG_EMBED_DIR").expect(
        "set DG_EMBED_DIR to a bundle folder with manifest.json, model.onnx, and tokenizer.json",
    ));
    let directory = directory.canonicalize().expect("DG_EMBED_DIR does not exist");
    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(directory.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    for name in ["manifest.json", "model.onnx", "tokenizer.json"] {
        println!("cargo:rerun-if-changed={}", directory.join(name).display());
    }
    for name in ["model.onnx", "tokenizer.json"] {
        let expected = manifest["sha256"][name].as_str().expect("manifest lacks a hash");
        let mut file = fs::File::open(directory.join(name)).expect("open bundle file");
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1 << 20];
        loop {
            let count = file.read(&mut buffer).expect("read bundle file");
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        assert_eq!(format!("{:x}", hasher.finalize()), expected, "hash mismatch: {name}");
    }
    // include_str!/include_bytes! need forward slashes on every platform.
    let path = directory.to_str().expect("bundle path is not UTF-8").replace('\\', "/");
    let path = path.strip_prefix("//?/").unwrap_or(&path);
    println!("cargo:rustc-env=DG_EMBED_DIR={path}");
    println!("cargo:rustc-cfg=dg_embedded");
}
