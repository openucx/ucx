# UCX PR Review Guidelines

This file covers pull-request review behavior and comment style. Detailed code
ownership, style, build, and testing rules live in `AGENTS.md` files and the
project docs linked from the root guide.

---

## Review Philosophy

Good reviews are direct, specific, and short. Reviewers commonly ask a
question rather than making a demand — "can we ...?", "why not ...?" — which
signals a soft suggestion versus a hard requirement.

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
- **Don't pile on** — if the same issue is already raised in the PR
  discussion, do not raise it again; support the existing thread instead. If
  the author replied with a good explanation, move on unless the current code
  or CI contradicts it

---

## PR Size and Scope

Large PRs slow down the entire team. Enforce these hard limits:

- **More than 500 added lines** — not allowed; ask the author to split
- **Bug fix PRs** — must contain only the fix and directly related test
  coverage; no unrelated refactoring
- **Feature PRs** — must not refactor code unrelated to the feature
- **Refactoring** — must be in a separate PR, with no functional changes

---

## Backports

A backport applies a PR that is already merged to `master` to a release
branch, such as `v1.23.x`. Review it against the original PR:

- **Name the original PR** — the description should link the original PR,
  for example "Backport of #12246"; ask the author to add it if it is missing
- **Same changes** — compare the backport with the original PR file by file;
  every change of the original should be there, and nothing unrelated should
  be added
- **Adaptations** — a difference from the original is allowed only to fit the
  release branch, for example code that moved or a helper that does not exist
  there; it must not change behavior compared with the original PR, and the
  description should explain it
- **One PR at a time** — a backport should apply exactly one original PR;
  ask the author to split a backport that combines several PRs
- **Size** — the 500-added-line limit does not apply, because a backport
  keeps the size of the original PR
- **Tests** — tests that the original PR added or changed should come with
  the backport, adapted if needed

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
read any style docs relevant to the diff before commenting.
Flag missing or stale `AGENTS.md` guidance only when the PR introduces or
changes a durable recurring pattern, ownership boundary, or convention.

Review by risk, in order:

1. API/ABI and wire compatibility.
2. Correctness, error handling, cleanup, and resource lifetime.
3. Concurrency and hot-path performance.
4. Test expectations and build metadata for changed files.
5. Documentation and style.

Focused checks:

- For resource lifetime changes, verify ownership and cleanup scope before
  reporting a finding. For keyed or cached resources, trace the key/value scope
  end to end and check whether cleanup affects only the intended key, all
  matching keys, or unrelated live users.
- When a change marks, initializes, validates, or reclassifies an object/state,
  trace the object from acquisition to first use. The fix should be placed at
  the earliest point where the invariant becomes true, before any consumer can
  observe the old state.
- When a PR adds or changes log messages, check the level, the printed
  identifiers, and the message content against `docs/LoggingStyle.md`.
- When a PR adds or changes `AC_ARG_WITH`, `AC_ARG_ENABLE`, `AC_DEFINE`, or an
  `AM_CONDITIONAL`, trace the option through `configure.m4`, `Makefile.am`,
  and CI/build scripts. Check explicit `--with-*`/`--enable-*` failure paths,
  explicit `--without-*`/`--disable-*` paths, default behavior, and whether a
  relevant build job exercises the new behavior.
- When a PR changes code, documentation, or code comments, check that the
  documentation matches the behavior and that code comments match the
  implementation.

When a finding depends on a code rule, cite the rule from the relevant
`AGENTS.md`, `docs/CodeStyle.md`, `docs/LoggingStyle.md`, or
`docs/OptimizationStyle.md` instead of restating broad style guidance.

---

## Comment Style

| Situation | Phrasing pattern | Example |
|-----------|-----------------|---------|
| Hard requirement | Direct statement | `blocker: this will crash on NULL` |
| Standard request | Plain imperative, or "please" | `please define all variables at the top of the function` |
| Soft suggestion | `maybe` / `can we` | `maybe we could remove this wrapper?` |
| Opinion | `I think` | `I think it can be uint8_t and we limit to 256 channels` |
| Nit | `minor:` prefix | `minor: add a blank line before` |
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
never blockers — they go in the no-prefix or `minor:` bucket. An uncertain
concern is never a blocker.

**Length**: 1–2 sentences per comment. Use a longer comment only when the
issue is complicated and non-obvious.

**Language:** Use simple English — many contributors are non-native speakers.
Do not use abbreviations such as "pls", "lgtm", "imo", "idk", etc.

**Actionability:** Omit observations that identify neither a concrete impact
nor a specific question or requested action.

**Comment order:** For change requests, start with the requested action, then
briefly explain why it is needed when the reason is not obvious. For uncertainty
or intent, ask a direct question. Example:
`please add $(UCX_LT_RELEASE) here as well. Otherwise, libucs_signal stays
outside the suffix scheme.`

**Comment scope:** One comment per distinct issue; do not bundle unrelated
points.

**Comment placement:** Inline comments for file-level issues; general PR
comment only for PR-level concerns.

**Review body:** Avoid generic review-summary comments such as `Code Review`,
finding counts, severity counts, or `N findings posted inline`. Leave the
review body empty unless there is a real PR-level concern.

**Code suggestions:** Prefer a GitHub `suggestion` block
(` ```suggestion ... ``` `) when the requested change is obvious and
mechanical. If needed, add one short sentence after the suggestion explaining
why the change is needed. For anything requiring judgment, describe the problem
and let the author propose the fix.
