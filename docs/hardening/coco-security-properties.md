# UCX CoCo Hardening Security Properties

This document defines the security boundary for UCX CoCo hardening in the IB
and mlx5 transports.

## Claimed Properties

When `uct_ib_md_is_coco_hardened()` is true, CoCo hardening is intended to
protect local memory safety for UCX-owned control paths. The hardened path must
preserve object lifetime, descriptor ownership, queue bounds, slot bounds, and
non-CoCo behavior.

The device and firmware are not trusted to provide authoritative metadata for
control-object safety. DEVX/FW output is hostile. Private UCX request state is
the authority for descriptor identity, operation ownership, object lifetime,
queue placement, and completion interpretation.

Capability queries are availability hints, not security facts. A reported
device capability can decide whether a feature can be attempted, but it cannot
prove that untrusted device output is safe to consume without UCX-side checks.

## Non-Properties

CoCo hardening does not claim payload authenticity, RDMA READ data integrity,
AM payload integrity, atomic return integrity, or denial-of-service resistance.

Untrusted device behavior can still drop work, delay progress, corrupt payload
bytes, reorder externally visible effects within device semantics, or return
syntactically valid but malicious metadata. Hardening must contain those effects
so they do not become local control-object memory corruption or unauthorized
descriptor/object reuse.

## Policy Boundary

`uct_ib_md_is_cc_dma_bounce()` reports the raw device/parent-domain condition
used for CoCo DMA-bounce allocation paths. `uct_ib_md_is_coco_hardened()` is the
policy predicate that gates hardened transport behavior.

Non-CoCo behavior must remain unchanged. Any CoCo-specific rejection, feature
masking, or validation must be conditional on `uct_ib_md_is_coco_hardened()`.
