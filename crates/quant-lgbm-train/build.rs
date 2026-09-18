use std::{env, path::PathBuf};

fn main() {
    let explicit = env::var_os("LIGHTGBM_LIB_DIR").map(PathBuf::from);
    let conda = env::var_os("CONDA_PREFIX").map(|value| PathBuf::from(value).join("lib"));
    let fallback = PathBuf::from("/Users/alanmxy/anaconda3/envs/ml311/lib");
    let directory = explicit
        .or(conda)
        .filter(|path| path.join("lib_lightgbm.dylib").exists())
        .or_else(|| {
            fallback
                .join("lib_lightgbm.dylib")
                .exists()
                .then_some(fallback)
        })
        .expect("set LIGHTGBM_LIB_DIR to the directory containing lib_lightgbm.dylib");
    println!("cargo:rustc-link-search=native={}", directory.display());
    println!("cargo:rustc-link-lib=dylib=_lightgbm");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
    println!("cargo:rerun-if-env-changed=LIGHTGBM_LIB_DIR");
}
