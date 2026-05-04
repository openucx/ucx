# UCX PR Review Guidelines

Topics are ordered by how frequently they appear in actual reviews.

---

## Review Philosophy

Good reviews are direct, specific, and short. The team's comments are almost always 1–3 lines.
Reviewers commonly ask a question rather than making a demand — "can we ...?", "why not ...?" —
which signals a soft suggestion versus a hard requirement.

- **Match title and description to the code** — if the PR says "fix login bug" but also refactors
  the auth module, flag it
- **Compactness** — the implementation should be as small as possible; extract duplicate code into
  helpers, avoid redundant state, remove anything unused
- **Consistency** — changes must match existing code in style, naming, abstraction level, and philosophy
- **Optimize the hot path** — performance matters most where code runs frequently; call it out there
- **Don't pile on** — if a comment is already raised or the author replied with a good explanation,
  move on

---

## PR Size and Scope

Large PRs slow down the entire team. Enforce these hard limits:

- **More than 500 added lines** — not allowed; ask the author to split
- **Bug fix PRs** — must contain only the fix and its tests; no unrelated refactoring
- **Feature PRs** — must not refactor code unrelated to the feature
- **Refactoring** — must be in a separate PR, with no functional changes

---

## What to Look For

Work through these categories in order of importance:

---

### 1. API Design and Code Structure

#### Function naming
- Boolean-returning functions must use an `is_` or `has_` prefix.
  Prefer `uct_gdaki_is_dmabuf_supported()` over `uct_gdaki_check_umem_dmabuf()`.
- Function names must describe *what* they do, not *how*.
  Prefer `uct_cuda_ipc_get_remote_address()` over a vague wrapper name.
- Hardware-specific terms must not leak into higher abstraction layers.
  `port_speed` is too HW-specific for a UCP-layer variable; use something like `device_bandwidth`.
- Functions declared as returning `int` should return `0`/`1`, not `ucs_status_t`; and vice versa.

#### Function design
- Prefer functions over macros when the code involves any logic.
- If a function modifies global state without holding a lock, prefer making it return the computed
  value and let the caller assign it.

#### Stable public API (src/ucp/api/ucp.h, src/uct/api/uct.h and what they include)
- **API/ABI must never break**, under any circumstances.
- API structs must use `field_mask` as their first field (a bitmap of available fields), to allow
  future extensions without breaking callers.
- API-change PRs should include only the minimal non-API changes needed to make the code compile;
  all other changes belong in a separate PR.

#### Interface correctness
- Use `sizeof(variable)` rather than `sizeof(type)` so the size stays correct if the type changes:
  `we prefer using sizeof(variable) over sizeof(type) to ensure correctness in case variable definition changed`.
- When exposing a capability (e.g., dmabuf support), cache the result the first time it is computed
  (static variable or struct field); never re-query on every call.
- Only set output parameters (`*out`) when the function returns `UCS_OK`.

#### Struct and field layout
- Avoid storing a value in a struct field if it can be derived from other existing fields.
- Prefer a separate struct field over packing multiple values into one field in a non-obvious way.

---

### 2. Correctness

#### Logic and conditions
- Simplify complex boolean expressions; split a multi-condition `if` into simpler nested conditions
  with comments explaining each.
- `!= UCS_NO` and `!= UCS_YES` are not the same as `== UCS_YES` and `== UCS_NO` — be explicit
  about which tri-state you are testing.
- For config tri-state, preserve user intent: `UCS_YES` reports failures, `UCS_NO` skips silently,
  and `UCS_TRY` tries and falls back quietly.
- Enum bits and flags: document flag combinations (e.g., when FAILOVER mode is enabled, both
  `UCP_EP_INIT_ERR_MODE_FAILOVER` and `UCP_EP_INIT_ERR_MODE_PEER_FAILURE` must be set).
- When changing a protocol or data format: **always ask, does this break wire compatibility?**

#### Type and size safety
- Use the narrowest integer type appropriate (e.g., `uint8_t` when values are bounded to [0, 255])
  and add a bounds check at assignment time.
- Prefer bitwise operations over modulo when the divisor is a power of 2; ensure the value is
  actually a power of 2 first (round up or assert).

#### Invariants and assertions
- Use `ucs_assertv` (with format args) rather than bare `ucs_assert` whenever the condition
  references variables — this makes assertion failures debuggable.
- When a comment says "X must be enabled" or "Y must be non-NULL here", add a matching
  `ucs_assert`/`ucs_assertv` rather than a silent `if` check.
- For pointer parameters that must never be NULL (internal APIs), use `ucs_assert(ptr != NULL)`
  instead of a runtime null-check that silently returns.
- Treat impossible internal states as assertions, not silent recovery. If a runtime null-check only
  hides a bug, replace it with `ucs_assert`/`ucs_assertv` so the bug is loud.
- Initialize data structures unconditionally (e.g., always init an LRU list even if disabled),
  and add assertions on access — this gives clearer error messages than a segfault.

#### Completeness
- When handling a set of values (error codes, memory types, flags), ask: is this set complete?
  If `UCS_ERR_NO_RESOURCE` is skipped, should `UCS_ERR_NO_DEVICE` and `UCS_ERR_UNSUPPORTED` also
  be skipped?
- When fixing a fast path, check the sibling paths: if `put_short` is fixed, should
  `put_offload_short` be fixed too?
- When adding a new memory type or transport, check that all tests and utilities cover the new case.
- When changing a struct, check all serialization/deserialization paths, not just the modified one.

---

### 3. Code Duplication and Abstraction

The team has a strong preference for DRY code. When the same pattern appears in two or more places,
extract it — even if the PR author did not introduce the duplication.

- When the same multi-step sequence appears at multiple call sites, extract a helper:
  `"the pattern of X + Y is repeating in multiple places and is logically a single unit, can we make it a helper function?"`
- Reuse existing functions that already perform the needed operation, including their error logging:
  `"can we reuse uct_ib_reg_mr function that also prints the error message?"`
- Before adding a new utility, search the codebase — functions like `ucs_for_each_bit`,
  `mem_buffer`, `ucs::handle<>` often already exist.
- Common system bus topology logic belongs in the shared topology layer, not duplicated in
  component-specific code.
- A helper function that only wraps one call and adds nothing should be removed.

---

### 4. Error Handling

#### Error propagation
- All functions that can fail must return `ucs_status_t` (or `int` with a clear negative=error
  contract). Never swallow a status silently.
- Use the `goto err` pattern for cleanup on failure paths; group cleanup labels in reverse init order.

#### Error messages
- Every non-trivial `UCS_ERR_*` return path must log a `ucs_error("...")` message that includes
  the resource name, device name, and/or error code/string.
- Use `ucs_fatal` only for truly unrecoverable programming errors; use `ucs_error` + `return
  UCS_ERR_*` for runtime failures.
- When adding a new error code to a skip-list, check whether related codes should also be skipped.
- `UCS_ERR_NO_RESOURCE` means "try again later" (back-pressure); it must never be logged as an error.

#### Cleanup paths
- Each `goto err` label must be idempotent (safe to call even if the corresponding init step was
  not completed).
- Lock acquisition failures must always cause a function return, not a fall-through.

---

### 5. Memory Management

- Use UCX allocation primitives (`ucs_malloc`, `ucs_calloc`, `ucs_free`) rather than bare
  `malloc`/`free` — they integrate with memory tracking.
- After every allocation, check the return value for NULL and propagate the error.
- Zero-initialize structs when fields may be conditionally set and later read unconditionally.
- Match every `init` with a `destroy`/`cleanup` on all error paths, including partially-initialized
  structures.

---

### 6. Performance

*Reviewers pay close attention to hot-path overhead.*

#### Inlining
- Use `UCS_F_ALWAYS_INLINE` only for functions on the **critical / per-message path**.
  Functions called only during init or config do not benefit from forced inlining and add code-size
  overhead.
- If a function calls another that already inlines the hot logic, adding another inline layer is
  rarely useful.

#### Caching expensive results
- One-time queries (dmabuf support, CPU model, device capability) must be cached in a static
  variable or a persistent struct field; avoid re-query on every call.
- Avoid calling a function that already caches its result repeatedly when one call suffices.

#### Lock overhead
- Distinguish "safe" (lock-free, caller holds lock) from "unsafe" (takes its own lock) variants.
  Adding lock acquisition inside a function named `_unsafe` is a contradiction.
- Minimize lock scope: take the lock as late as possible and release it as early as possible.
- When a single mutex can replace two, prefer the simpler design unless contention is measured.

#### Hot path vs. slow path
- Code on initialization or the slow path (protocol selection, config parsing) must not add
  overhead to the fast path; it should live in `.c` files, not `.inl` files.
- Unconditionally incrementing/decrementing a counter on the hot path can be cheaper than a branch,
  even if the counter is not always used — profile before adding branches.

---

### 7. Concurrency and Thread Safety

- Every shared data structure access must either hold the appropriate lock or be explicitly
  documented as "protected by caller" in the function comment.
- If a function comment states a lock requirement, add a `ucs_assert(ucs_spin_is_locked(...))` to
  make it machine-verifiable.
- When combining two locks, always acquire them in the same order everywhere; document the order.
- Use atomic operations for reference counts and flags accessed from multiple threads.
- Use `ucs_recursive_spinlock` only when the same thread can re-enter; otherwise use a plain spinlock.
- Operations on UCT interfaces and worker resources must only be called from the worker context
  (the thread that owns the worker, while holding the worker lock).
- Functions called from async/signal context must not allocate memory or take blocking locks.

---

### 8. Tests

#### What to test
- Every new feature must have a gtest in `test/gtest/` covering the happy path and at least one
  failure or edge case.
- Every bug fix must include a regression test that would have caught the bug before the fix.
- Before adding a component-specific test, ask: can the existing generic infrastructure
  (`test_md`, `mem_buffer`, `rkey_ptr`) be extended to cover the use-case instead?

#### Writing good tests
- Use `mem_buffer` for generic memory allocation/comparison across all memory types:
  `ucs_for_each_bit(mt, md_attr().reg_mem_types | md_attr().alloc_mem_types)`.
- Use `ucs::handle<T>` and `UCS_TEST_CREATE_HANDLE` for RAII in C++ tests.
- Test constants must reference the source constant, not be hardcoded:
  use `UCS_LOG_MULTILINE_OUTPUT_SIZE` rather than `4096`.
- For retry/pending behavior, use a polling loop rather than fixed iterations or sleeps:
  `do { progress(); status = ...; } while (status == UCS_ERR_NOT_CONNECTED)`.

---

### 9. Logging

*The team is very specific about log levels and message quality.*

#### Log level selection
| Level | When to use |
|-------|-------------|
| `ucs_debug` | Routine informational messages on init/progress path |
| `ucs_diag` | Unexpected soft failure: system still works with degraded capability |
| `ucs_warn` | Unexpected condition the code can recover from |
| `ucs_error` | Operation failed; caller will see an error status |
| `ucs_fatal` | Unrecoverable programming error; process must exit |

- Use `ucs_diag` (not `ucs_error`) only for unexpected but recoverable conditions where UCX falls
  back gracefully.
- Use `ucs_error` (not `ucs_debug`) for genuine failures that the user must investigate.

#### Message content
- Messages must be lower-case (except acronyms).
- Include the relevant context: device name, index, error code/string, size, etc.
- Be specific: `"SHM_HUGETLB is not supported on the system"` not `"huge pages not supported"`.
- Remove dead `ucs_debug` / `printf` calls left from development; replace or delete them.

---

### 10. Naming

- Output pointer parameters: `_p` suffix (e.g., `local_sys_dev_p`).
- Match the capitalization, underscore style, and prefix of surrounding code.
- When existing similar code uses `_map` (e.g., `lane_map`), don't introduce `_mask` for a
  conceptually identical thing in the same scope.
- When renaming, update all uses including comments and log messages.

---

### 11. Code Style and Formatting

*Usually `minor:` severity.*

- **Variable declarations**: in C code, declare all local variables at the top of the function,
  before any statements. Within that block, initialized declarations (e.g., `int x = 0;`) come
  before uninitialized ones (e.g., `int ret;`).
- **Blank lines**: one blank line between logical sections; no blank lines inside struct
  definitions or between consecutive declarations.
- **Trailing whitespace**: none.
- **Double negation**: avoid `!!expr`; use `(expr) != 0` or a properly named `is_` function.
- **TODO comments**: make them specific (`/* TODO: <task> */`) or convert to a tracked issue;
  remove vague or stale ones.
- **Dead code**: remove commented-out code, unused variables, and unreachable functions.
- **UCG references**: UCG is being removed; remove any remaining references to it.

---

### 12. Documentation and Comments

- Public functions in headers should have a short doxygen-style comment describing parameters and
  return value.
- Non-obvious logic (e.g., `!= UCS_NO` instead of `== UCS_YES`) deserves a one-line comment
  explaining why.
- When implementing a protocol detail, cite the spec section or describe the protocol behavior.
- Significant user-visible changes (new features, deprecations, behavior changes) must have an
  entry in the `NEWS` file.

---

### 13. Build and Configuration

- New config entries must have descriptive help strings explaining allowed values and their effect.
- `configure.ac` / `m4` macros for optional libraries must check minimum versions.
- Avoid adding build-system conditionals for features that are always available.
- Script-based filters should derive paths from authoritative sources (`.gitmodules`) rather than
  hardcoded lists.

---

## Comment Style

| Situation | Phrasing pattern | Example |
|-----------|-----------------|---------|
| Hard requirement | Direct statement | `blocker: this will crash on NULL` |
| Standard request | `pls` + imperative | `pls define all vars at the top of the func` |
| Soft suggestion | `maybe` / `can we` | `maybe we could remove this wrapper?` |
| Opinion | `IMO` | `IMO it can be uint8_t and we limit to 256 channels` |
| Nit | `minor:` prefix | `minor: pls add blank line before` |
| Question about intent | Direct question | `does it break wire compatibility?` / `why is it needed?` |
| Alternatives | `why not` / `instead` | `instead of X, can we use Y?` |
| Code suggestion | Inline block | ` ```suggestion ... ``` ` |

**Severity levels:**

| Prefix | Meaning |
|--------|---------|
| `blocker:` | Must fix before merge: crash, data corruption, security hole, broken CI |
| *(none)* | Should fix; warrants discussion if there is a good reason not to |
| `minor:` | Nice to fix, will not block merge |

**Blocker bar is high.** Config naming debates, default value choices, and documentation gaps are
never blockers — they go in the no-prefix or `minor:` bucket.

**Length**: 1–2 sentences per comment. Use simple English — many contributors are non-native
speakers. Lead with the *problem* (`"this can be NULL here"`) not the solution (`"you should add a
null check"`). Skip suggested code blocks unless the fix is non-obvious. One comment per distinct
issue; do not bundle unrelated points. Inline comments for file-level issues; general PR comment
only for PR-level concerns.

Use the GitHub `suggestion` block (` ```suggestion `) when the fix is mechanical and unambiguous.
For anything requiring judgment, describe the problem and let the author propose the fix.
