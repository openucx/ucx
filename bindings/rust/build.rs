use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=../../src/ucp/.libs/");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=ucp");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // Some of the UCX detailed examples in comments can confuse the
        // bindgen parser and it will make bad code instead of comments
        .generate_comments(false)
        // ucs_status_t is defined as a packed enum and that will lead to
        // badness without the flag which tells bindgen to repeat that
        // trick with the rust enums
        .rustified_enum(".*")
        .clang_arg("-I../../src/ucp/api/")
        .clang_arg("-I../../")
        .clang_arg("-I../../src/")
        // Annotate ucs_status_t and ucs_status_ptr_t as #[must_use]
        .must_use_type("ucs_status_t")
        .must_use_type("ucs_status_ptr_t")
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
