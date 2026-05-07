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
- **Bug fix PRs** — must contain only the fix and its tests; no unrelated refactoring
- **Feature PRs** — must not refactor code unrelated to the feature
- **Refactoring** — must be in a separate PR, with no functional changes

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
4. Tests and build metadata for changed files.
5. Documentation and style.

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

**Blocker bar is high.** Config naming debates, default value choices, and documentation gaps are
never blockers — they go in the no-prefix or `minor:` bucket.

**Length**: 1–2 sentences per comment. Use simple English — many contributors are non-native
speakers. Lead with the *problem* (`"this can be NULL here"`) not the solution (`"you should add a
null check"`). Skip suggested code blocks unless the fix is non-obvious. One comment per distinct
issue; do not bundle unrelated points. Inline comments for file-level issues; general PR comment
only for PR-level concerns.

Use the GitHub `suggestion` block (` ```suggestion ... ``` `) when the fix is
mechanical and unambiguous.
For anything requiring judgment, describe the problem and let the author propose the fix.
