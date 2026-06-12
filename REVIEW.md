# UCX PR Review Guidelines

This file covers pull-request review behavior and comment style. Detailed code
ownership, style, build, and testing rules live in `AGENTS.md` files and the
project docs linked from the root guide.

---

## Review Philosophy

Good reviews are direct, specific, and short. The team's comments are almost always 1–3 lines.
Reviewers commonly ask a question rather than making a demand — "can we ...?", "why not ...?" —
which signals a soft suggestion versus a hard requirement.

- **Match title and description to the code** — the description should follow
  `.github/PULL_REQUEST_TEMPLATE.md`, the "Why?" should explain the value for
  the user, and stale or missing major changes should be flagged with a
  suggested update
- **Compactness** — the implementation should be as small as possible; avoid
  redundant state and remove anything unused
- **Consistency** — changes must match existing code in style, naming,
  abstraction level, and philosophy
- **Optimize the hot path** — prioritize performance comments when the changed
  code runs frequently
- **Don't pile on** — if the same issue is already raised in an unresolved
  thread, do not raise it again; support the existing thread instead. If the
  author replied with a good explanation, move on

---

## PR Size and Scope

Large PRs slow down the entire team. Enforce these hard limits:

- **More than 500 added lines** — not allowed; ask the author to split
- **Bug fix PRs** — must contain only the fix and directly related test
  coverage; no unrelated refactoring
- **Feature PRs** — must not refactor code unrelated to the feature
- **Refactoring** — must be in a separate PR, with no functional changes

---

## Test Expectations

Derive required test coverage first from the PR's stated goal, then from the
behavioral risks introduced by the implementation. Avoid asking for tests that
cover incidental implementation details without a plausible regression risk.

- **Bug fixes** — should usually add or update a focused regression test,
  because the existing suite did not catch the issue. If a test is not
  practical, ask the author to explain the coverage gap
- **New APIs** — should usually include a test that exercises the new API
  contract, including error handling or compatibility behavior when relevant
- **Internal flow changes** — should be covered by existing tests/CI or by a
  new focused test. If relying on existing coverage, identify the relevant test
  or CI job before asking for more tests
- **Build/CI changes** — should have enough validation to show the stated CI
  behavior changed as intended, either through CI results, dry-run output, or a
  small script/unit test when practical
- **Configure/build options** — new or changed user-facing configure flags
  should usually add CI coverage for the explicit enable/disable path that
  changed. In UCX this often belongs in `buildlib/tools/builds.sh` or
  `contrib/test_jenkins.sh`; for disable options, also verify the disabled
  module, object, or generated artifact is absent when practical

---

## Review Checklist

Apply the `AGENTS.md` discovery rule from the root guide for changed paths, and
read any style docs relevant to the diff before posting findings.
Check whether changed folders need `AGENTS.md` updates: add guidance for new
recurring patterns, remove stale local guidance, and do not leave ownership
gaps.

Review by risk, in order:

1. API/ABI and wire compatibility.
2. Correctness, error handling, cleanup, and resource lifetime.
3. Concurrency and hot-path performance.
4. Test expectations and build metadata for changed files.
5. Documentation and style.

Focused checks:

- For resource lifetime changes, verify ownership and cleanup scope before
  posting a finding. For keyed or cached resources, trace the key/value scope
  end to end and check whether cleanup affects only the intended key, all
  matching keys, or unrelated live users.
- When a change marks, initializes, validates, or reclassifies an object/state,
  trace the object from acquisition to first use. The fix should be placed at
  the earliest point where the invariant becomes true, before any consumer can
  observe the old state.
- When a PR adds or changes `AC_ARG_WITH`, `AC_ARG_ENABLE`, `AC_DEFINE`, or an
  `AM_CONDITIONAL`, trace the option through `configure.m4`, `Makefile.am`,
  and CI/build scripts. Check explicit `--with-*`/`--enable-*` failure paths,
  explicit `--without-*`/`--disable-*` paths, default behavior, and whether a
  relevant build job exercises the new behavior.

When a finding depends on a code rule, cite the rule from the relevant
`AGENTS.md`, `docs/CodeStyle.md`, `docs/LoggingStyle.md`, or
`docs/OptimizationStyle.md` instead of restating broad style guidance.

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

**Severity formatting:** Do not use markdown severity headings, emoji labels,
`[P2]`, `[important]`, or uppercase labels such as `MINOR`/`BLOCKER`. If
severity is needed, use only the inline prefixes from the table: `blocker:` or
`minor:`.

**Blocker bar is high.** Config naming debates, default value choices, and documentation gaps are
never blockers — they go in the no-prefix or `minor:` bucket.

**Length**: 1–2 sentences per comment. Use a longer comment only when the
issue is complicated and non-obvious.

**Language:** Use simple English — many contributors are non-native speakers.

**Comment order:** Start with the requested change, then briefly explain why it
is needed when the reason is not obvious. Example:
`pls add $(UCX_LT_RELEASE) here as well; otherwise libucs_signal stays outside
the suffix scheme.`

**Comment scope:** One comment per distinct issue; do not bundle unrelated
points.

**Comment placement:** Inline comments for file-level issues; general PR
comment only for PR-level concerns.

**Review body:** Avoid generic review-summary comments such as `Code Review`,
finding counts, severity counts, or `N findings posted inline`. Leave the
review body empty when inline comments are self-contained. Use a top-level PR
comment only for a real PR-level concern.

**Code suggestions:** Prefer a GitHub `suggestion` block
(` ```suggestion ... ``` `) when the requested change is obvious and
mechanical. If needed, add one short sentence after the suggestion explaining
why the change is needed. For anything requiring judgment, describe the problem
and let the author propose the fix.
