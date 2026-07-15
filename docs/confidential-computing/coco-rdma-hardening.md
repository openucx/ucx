# UCX Confidential Computing RDMA Hardening

This document summarizes the UCX InfiniBand/mlx5 hardening used when UCX runs
with a Confidential Computing RDMA device whose NIC-visible control memory is
shared with an untrusted device.

## Scope

The hardening applies only when UCX selects the CoCo policy predicate:

```c
uct_ib_md_is_coco_hardened(md)
```

The raw device condition is reported separately by
`uct_ib_md_is_cc_dma_bounce()`. The policy predicate gates transport behavior;
the raw predicate gates allocation/control-memory mechanics. Non-CoCo devices
must keep the normal UCX path: no CoCo feature filtering, no CoCo-only shadow
state, no CoCo CQE validators, and no changed transport discovery.

## Security Properties

When CoCo hardening is active, UCX treats NIC/firmware-controlled metadata as
hostile until it is checked against private UCX state. The implementation is
intended to protect local UCX control-object memory safety, descriptor
ownership, queue and slot bounds, object lifetime, and QP/CQ/SRQ binding.

The hardened path does not authenticate application payload bytes or remote
memory contents. RDMA READ data, active-message payloads, immediate data, and
atomic return values remain user payload. Denial of service is also outside the
security property: the device can still drop, delay, or fail work, but it must
not be able to turn forged metadata into unsafe local control-object access.

## Supported CoCo Profile

The audited CoCo DEVX profile is intentionally narrow:

- RC mlx5 transport.
- RC QPs.
- CQ objects created through the CoCo-safe control path.
- Cyclic SRQ/RMP only with private SRQ slot shadow state and validation.
- UMEM/MR/MKey objects whose bounds and permissions are recorded privately.

The CoCo path rejects or hides unaudited features before traffic starts. This
includes non-RC transports such as UD and DC, non-RC QP modes, tag matching,
CQE compression/zipping, scatter-to-CQE, CQ resize, DDP/OOO/AR, UMR, indirect
mkeys, signature/T10-DIF, and unaudited DEVX object types.

## Private UCX State

Device-visible control memory is not authoritative. UCX keeps private state for
the fields needed to validate completions and object lifetimes:

- CQ, QP, SRQ/RMP, UMEM, MR, and MKey registries.
- QP number plus generation/state and CQ/iface binding.
- TX shadow entries for posted operations and expected completions.
- SRQ slot state, generation, descriptor pointer, and posted receive length.
- Requested UMEM/MR address ranges and permissions.

Device-provided identifiers such as CQN, QPN, RMPN, UMEM id, lkey, and rkey
become trusted only after UCX records them in the matching private registry and
verifies that they do not collide with an existing live object.

## DEVX Output Validation

CoCo DEVX create/query results are parsed through guarded helpers. UCX checks
the output length before reading fields, rejects duplicate object identifiers,
keeps requested sizes and permissions authoritative, and rejects device output
that widens memory ranges or grants unexpected access.

For memory registration and MKeys, UCX records the private address range,
length, lkey/rkey, and access flags. Later use of those keys is valid only
within the recorded bounds and permissions.

## CQE Validation

CQEs are consumed in a CoCo-specific RC mlx5 path. After the normal ownership
load barrier, UCX snapshots the fields it will use and validates them before
touching descriptors or changing private state.

TX completions are accepted only if the opcode, QPN, CQ binding, WQE counter,
and generation match a live private TX shadow entry.

RX completions are accepted only if the opcode and format are supported, the QPN
matches a live private QP/SRQ attachment, the SRQ slot is posted for the current
generation, and `byte_cnt` is within the private posted receive length.

Error CQEs are treated as untrusted selectors. UCX maps them to private TX or
RX/SRQ state before retiring descriptors. Syndrome and vendor syndrome fields
are diagnostic only and never authorize object ownership.

## Taint And Validation Contract

The following table summarizes how common device-originated values may be used
after validation.

| Input | Source | UCX validation before use | Resulting contract |
| --- | --- | --- | --- |
| CQE opcode and format bits | CQE | Accepted only by the CoCo TX/RX/error validator for the selected path. | May select the validated private-state check; unsupported formats are rejected. |
| QPN | CQE or DEVX output | Masked, then checked against private QP registry, generation/state, CQ, iface, and object class. | Lookup key only until private registry validation succeeds. |
| WQE/SRQ counter | CQE | Checked against live private TX shadow or SRQ slot state and generation. | Trusted only after range, liveness, and generation checks. |
| `byte_cnt` | RX CQE | Bounded by private posted receive length before callback construction. | Bounded payload length; cannot exceed the private posted length. |
| CQN/QPN/RMPN/UMEM/MKey ids | DEVX output | Output length checked, object class checked, duplicate ids rejected, private registry populated before publication. | Trusted only as keys to matching private registry records. |
| MR/MKey length and access | DEVX or verbs output | Compared against UCX-requested range and permissions; widened ranges or permissions are rejected. | Trusted only within private recorded bounds. |
| Syndrome/vendor fields | Error CQE or DEVX output | No ownership validation; used only after private object mapping if needed. | Diagnostic only. |
| AM payload, immediate data, RDMA READ data, atomic return values | Payload/device data | Descriptor ownership and length may be checked, but contents are not authenticated by this hardening. | User payload, not protected by the CoCo control-object hardening. |

## Violation Handling

Forged, stale, duplicated, out-of-range, or unsupported device metadata is
fail-closed. The affected CQ, QP, SRQ/RMP, interface, or memory object is
poisoned or rejected, and normal polling/posting on the affected object stops.
Violation handling is kept on a cold path; successful polling and posting do
not update diagnostic counters or logs.

## Non-CoCo Behavior

The hardening policy must not activate for non-CoCo memory domains. Normal UCX
transport discovery, selected operations, allocation behavior, DEVX behavior,
and SRQ/RMP behavior remain unchanged when `uct_ib_md_is_coco_hardened(md)` is
false.
