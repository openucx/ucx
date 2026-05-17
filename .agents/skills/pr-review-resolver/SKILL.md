---
name: "pr-review-resolver"
description: Walks the unresolved review comments on a UCX GitHub PR and proposes a code change per comment. Use when the user asks to address PR review comments, go through a PR review, fix review feedback, resolve reviewer feedback, apply suggestion blocks, or work through pull-request review comments on an open GitHub PR.
---

# PR Review Resolver

## Overview

Resolves unresolved review comments on a GitHub PR by classifying each comment using a two-tier classifier (auto-tier batched, manual-tier individual), proposing a concrete change, asking the user for approval, applying the approved edits, and finalizing with one commit + push only when every comment was addressed. Comments that the user skips leave the working tree dirty for the user to finish manually. Requires the gh CLI authenticated and the user already on the PR branch.

## When to Use This Skill

Use this skill when the user asks something like:

- "Address the review comments on PR 1234"
- "Go through the PR review"
- "Fix the review feedback"
- "Resolve the reviewer comments"
- "Apply the suggestions from this review"
- "Go through what the reviewer asked for"

Do **not** use it for: writing a fresh review of someone else's PR, reviewing your own diff before pushing, replying on GitHub without making edits, or working with GitLab merge requests.

## Prerequisites

- `gh` CLI installed and authenticated (`gh auth status` succeeds).
- `git` available on `PATH`.
- The user is on a checked-out PR branch in a local clone of the repo the PR belongs to.
- The PR is hosted on GitHub.
- Clean working tree. If working tree is dirty, print the dirty files and ask "proceed and co-mingle changes in the eventual commit, or abort?" Wait for the user. If user abort, return immediately and do not start the workflow

## Workflow

The skill executes the steps below in order. Each step has a clear gate; stop and surface the issue if a gate fails.

### Step 1 — Prerequisite gate

Run these checks before doing anything else. If either fails, print the failure with a remediation hint and exit. Do not call `gh api`, do not read files, do not edit anything.

```bash
command -v gh >/dev/null 2>&1 || { echo "gh CLI not found. Install with: curl -sS https://webi.sh/gh | sh"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated. Run: gh auth login"; exit 1; }
```

If a prereq is missing, the entire run aborts here. There is no fallback path.

### Step 2 — Resolve the PR

Resolution priority:

1. **Explicit reference in the user's prompt.** A GitHub PR URL (`https://github.com/<owner>/<repo>/pull/<n>`) or `#<n>` wins.
2. **Current branch.** Run `gh pr view --json number,url,baseRefName,headRefName,headRepository`. Use that PR.
3. **Neither.** Ask the user once for a PR URL or number, then exit if they don't provide one.

Echo the resolved PR number + URL back to the user before fetching anything else, so they can confirm the target.

### Step 3 — Fetch unresolved review comments

Use the GraphQL API to filter to **unresolved** review threads (the REST `/pulls/{n}/comments` endpoint does not expose the resolution state directly):

```bash
gh api graphql -f query='
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        reviewThreads(first: 100) {
          nodes {
            isResolved
            comments(first: 50) {
              nodes {
                id
                databaseId
                author { login }
                path
                line
                originalLine
                body
              }
            }
          }
        }
        reviews(first: 50) {
          nodes {
            author { login }
            submittedAt
            state
            body
          }
        }
      }
    }
  }' -F owner=<owner> -F repo=<repo> -F number=<n>
```

From the response:

- Keep only `reviewThreads.nodes` where `isResolved == false`. For each kept thread, take the **first comment** in the thread (the originating comment) as the actionable comment; treat any later comments in the thread as reply context.
- Keep `reviews.nodes` where `body` is non-empty and the body is not a duplicate of one of the inline-comment bodies. These are the top-level review summaries — surfaced at end-of-run as non-actionable reminders.

Normalize each actionable comment to: `{ id, author, file, line, body, has_suggestion_block, reply_context }`.

`has_suggestion_block` is true when the body contains a ` ```suggestion ` fenced block.

### Step 4 — Classify each comment

For every actionable comment, apply this rubric **before** the user sees anything. A comment is **auto-tier** only when all five "auto" cells hold; otherwise it is **manual-tier**. GitHub suggestion blocks are not auto-promoted — they go through the same rubric like any other comment.

| Signal | Auto-tier | Manual-tier |
|---|---|---|
| File type | `.md`, `.rst`, `.txt`, or comment-only lines inside `.c` / `.h` / `.cpp` / `.py` | source lines in `.c` / `.h` / `.cpp` / `.py` (non-comment) |
| Change kind | typo, wording, formatting, doc link, comment rewrite | logic, API signature, control flow, error handling, memory / locking / RDMA semantics, build files |
| Reviewer ask | direct textual instruction ("fix typo", "rename `X` to `Y`") | open-ended ("consider…", "what if…?", "could leak / race / deadlock") |
| Scope | ≤ ~3 lines, single hunk, no cross-file rename of an exported symbol | multi-file, cross-function, or any rename of an exported symbol |
| Test impact | none (build + test outcome unchanged) | could change build, runtime, or test results |

**Default to manual-tier whenever any signal is ambiguous.** Friction is cheaper than a silently-applied wrong fix.

Edge cases that always force manual-tier regardless of the rubric:

- Comment anchor is stale (line no longer exists at the cited file:line; check via `git ls-tree HEAD -- <file>` and `awk` the line if needed). Surface as "anchor stale — please confirm target".
- Suggestion block fails a literal in-place apply because surrounding lines drifted. Surface with a "fuzzy match" indicator.
- Two reviewers leave contradictory comments on the same `file:line`. Surface both together and let the user pick.

### Step 5 — Auto-tier batch approval

If there is at least one auto-tier comment, present them as a single bulleted summary, one line per change. Format:

```
Auto-tier batch (N changes):
  1. <file>:<line> — <comment author>: <one-line "old → new" summary>
  2. ...
Approve all / Skip all / Approve all except <comma-separated indices>?
```

Accept these responses:

- `approve` / `approve all` / `yes` → apply every change.
- `skip` / `skip all` / `no` → record each as a skipped comment with reason "user declined batch".
- `approve except 2, 4` → apply the rest; record 2 and 4 as skipped with reason "user excluded from batch".
- Free-form text → ask one clarifying yes/no.

For each approved change, edit the file in the working tree. Do **not** stage or commit at this step.

### Step 6 — Manual-tier walk

For each manual-tier comment, first print this four-block context, then gate the decision with the **AskUserQuestion tool** so the user can pick with arrow keys / number keys instead of typing free-form text:

```
--- Comment <i> of <N manual> ---
Author:    <login>
Location:  <file>:<line>
Comment:   <body>

Proposed change:
<diff or before/after block>

Reasoning:
<one sentence explaining why this addresses the comment>

Risk note:
<one sentence flagging what could regress, callers that may need attention,
 or "no expected risk — change is local to this function">
```

Then call `AskUserQuestion` with:

- `question`: `"How to handle this comment?"`
- `header`: `"Decision"`
- `multiSelect`: `false`
- `options` (in this order, so option 1 is the default-accept):
  1. **Approve** — Apply the proposed change as shown.
  2. **Edit** — Apply a user-revised version of the change.
  3. **Skip** — Leave the file untouched and record a reason.

Handle the selection:

- **Approve** → apply the proposed change to the working tree.
- **Edit** → follow up with a normal text prompt: `"Paste the revised change (diff or full replacement text):"`. Apply the user's revision instead of the original proposal.
- **Skip** → follow up with a normal text prompt: `"Reason for skipping?"`. Do not edit; record the reason verbatim.
- **Other** (the free-form fallback AskUserQuestion always offers) → treat the typed text as a clarifying instruction; re-issue the AskUserQuestion once the intent is clear.

Apply edits in the working tree only — no staging, no commits at this step.

### Step 7 — End-of-run summary

After the last comment is decided, print:

```
=== PR <n> resolver summary ===

Applied: <count>
  - <file>:<line> — <one-line description>
  - ...

Skipped: <count>
  - <file>:<line> — reason: <user's recorded reason>
  - ...

Reviewer overall feedback (not auto-actionable — please address separately):
  <review summary 1, verbatim>
  <review summary 2, verbatim>
```

`applied + skipped` must equal the total ingested actionable comments. If they don't, surface the discrepancy — do **not** proceed to finalization.

### Step 8 — Finalization

**Rule:** commit + push **only** when the skipped count is zero.

If `skipped == 0`:

1. Get the UCX module prefix from the PR HEAD's most recent commit:

   ```bash
   last_subject=$(git log -1 --pretty=%s)
   prefix=${last_subject%%:*}
   ```

   If `prefix == last_subject` (i.e. no `:` in the subject), **stop**. Print: "Last commit on PR branch has no `MODULE:` prefix — please provide the prefix to use." Wait for the user; treat the user's response as the prefix. If the user declines, leave the working tree dirty with no commit.

2. Stage and commit:

   ```bash
   git add -u
   git commit -m "${prefix}: PR fixes from agent"
   ```

3. Push to the PR's head ref:

   ```bash
   git push
   ```

   If `git push` fails (non-fast-forward, branch protection, etc.), surface the error verbatim and instruct the user to resolve and re-run. Do **not** attempt force-push.

4. Print the new commit SHA.

If `skipped > 0`:

```
Leaving working tree dirty; <N> comment(s) still need your attention.
No commit was created and no push was attempted.
```

Exit without staging.

## Commit message convention

UCX uses subject-line module prefixes:

```
MODULE/UNIT1/UNIT2/...: <summary>
```

This skill always uses the literal summary `PR fixes from agent` and copies the `MODULE/UNIT...` prefix (everything before the first `:`) from `git log -1 --pretty=%s` on the PR HEAD.

Examples:

- Last commit subject `ucp_proto/single/short: support new opcode` → agent commit `ucp_proto/single/short: PR fixes from agent`.
- Last commit subject `tools/info: print thread mode` → agent commit `tools/info: PR fixes from agent`.
- Last commit subject `ucp_proto, ucp_wireup: refactor handshake` → agent commit `ucp_proto, ucp_wireup: PR fixes from agent` (the whole prefix is copied verbatim).

## Edge cases

- **No unresolved comments.** Print "Nothing to do — PR has no unresolved review comments" and exit. Do not commit.
- **PR not found / no PR for current branch.** Ask once for a URL or number; abort if none provided.
- **`gh api` returns paginated results truncated at the 100 / 50 limits.** Note the truncation in the end-of-run summary and continue with what was fetched.
- **Mid-run failure** (network drop, agent host crash). Re-running is safe: already-applied edits remain on disk; re-ingestion will re-classify and re-prompt only for the still-unresolved threads.

## Out of scope

- Replying to comments on GitHub or marking threads resolved via the API.
- Force-push, rebase, branch creation, opening a new PR.
- Running tests, builds, or linters before pushing.
- Comments outside the formal review (general PR conversation thread, commit-line comments).
- GitLab merge requests.

## Additional Resources

- [gh CLI manual](https://cli.github.com/manual/)
- [GitHub PR review API (GraphQL)](https://docs.github.com/en/graphql/reference/objects#pullrequest)
- UCX commit conventions — see top of the UCX repo `CONTRIBUTING.md`.
