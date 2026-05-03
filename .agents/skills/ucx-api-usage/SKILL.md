---
name: ucx-api-usage
description: Guidance for writing applications, examples, bindings, or integration code on top of UCX public APIs. Use when asked how to use UCP, UCT, UCS, JUCX, Go bindings, examples, endpoint/worker/context setup, request lifetimes, progress, memory registration, or UCX API caveats.
---

# UCX API Usage

## Overview

Use this skill when writing or reviewing code that consumes UCX public APIs
rather than modifying UCX internals.

## Sources

Start with:

- `examples/AGENTS.md`
- `bindings/AGENTS.md` when touching language bindings
- Public headers under `src/ucp/api`, `src/uct/api`, `src/ucs`, and `src/ucm/api`
- Existing examples under `examples/`
- API documentation in `docs/` and public-header Doxygen comments

## API Usage Rules

- Treat public API structs as versioned contracts. Initialize them fully, set
  the `field_mask`, and only read fields that the corresponding mask exposes.
- Keep context, worker, endpoint, memory handle, request, and listener lifetimes
  explicit. Match every init/create/open with the documented close/destroy/free.
- Drive progress deliberately. UCP operations often require calls to progress
  functions until completion callbacks fire or requests complete.
- Handle `UCS_INPROGRESS` separately from success and failure. Do not treat it
  as a completed operation.
- Release non-NULL request handles using the API-prescribed free path after
  completion.
- Do not assume a transport, device, or memory type is available. Query
  capabilities or use UCX configuration instead of hardcoding hardware behavior.
- Use UCX allocation, memory registration, and rkey APIs according to the layer
  being used. Do not mix UCP and UCT handles unless the API explicitly permits
  it.
- Keep example code concise and focused on the API sequence being demonstrated.

## Caveats To Check

- Thread mode requested versus thread mode provided.
- Endpoint error handling mode and close semantics.
- Callback ownership and whether user data remains valid until completion.
- Packed remote keys and addresses must be released with the matching UCX API.
- Config objects must be released after use.
- Tests and examples should avoid requiring special hardware unless that is the
  subject of the example.

## Verification

For examples, build the normal tree after configuration. For bindings, use the
commands in `bindings/AGENTS.md` and report missing optional dependencies such
as Go, Java, Maven, CUDA, ROCm, or transport hardware.
