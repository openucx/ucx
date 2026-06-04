# Agent Guide: bindings

Language bindings over the UCP C API. Each binding is independent and
optional; both consume the installed `libucp` headers and shared library.

## Subdirectory Map

- `go/` — Go binding via cgo.
  - `src/ucx/` — the binding itself. `context.go`, `endpoint.go`,
    `listener.go`, `worker.go` each pair with a `*_params.go` sibling;
    other files (`am_data.go`, `callbacks.go`, `connection_request.go`,
    `memory.go`, `mmap_params.go`, `request.go`, `ucx_error.go`,
    `utils.go`) stand alone. The C shim header is `goucx.h`, paired
    with generated constant tables (`ucp_contsants.go`,
    `ucs_constants.{c,go,h}`).
  - `src/cuda/` — optional GPU helpers.
  - `src/examples/` — runnable example programs.
  - `tests/` — Go test suite (`*_test.go`); use `go test ./...` from
    `bindings/go/tests` after a UCX install.
- `java/` — Java binding via JNI.
  - `src/main/java/` — Java public API.
  - `src/main/native/` — JNI shim translating to UCP calls (has its own
    `Makefile.am`).
  - `src/test/` — JUnit tests.
  - Build is Maven-driven (`pom.xml.in` is templated by autoconf).
  - Style: `checkstyle.xml`. See `bindings/java/README.md` for build/run
    instructions.

## Conventions

- Bindings consume the public API only (`ucp/api/ucp.h`, `ucp_def.h`,
  `ucp_compat.h`, optionally experimental `ucpx.h`). Pulling internal
  headers from `core/`/`proto/` is a bug — those aren't ABI-stable.
- New UCP API additions need to be mirrored manually in each binding.
  The Go constant tables (`ucp_contsants.go`, `ucs_constants.*`) and the
  JNI signature glue have to track new enums/flags.
- Memory ownership across the cgo/JNI boundary follows UCP semantics:
  callbacks fire on the worker progress thread, so the binding must
  marshal completions back to the language runtime safely. Both
  bindings serialize through their own request handles (`request.go`
  for Go, `UcxCallback` for Java).
- Examples and tests should keep working against an installed UCX, not
  an in-tree build — they're how downstream users learn the bindings.

## Pointers

- UCP public API: `src/ucp/api/`.
- Build integration: each binding has its own `Makefile.am`
  (`bindings/go/Makefile.am`, `bindings/java/Makefile.am`); they are
  pulled in from the top-level `Makefile.am`.
- Java packaging helpers: `buildlib/jucx/`.
