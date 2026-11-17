---
name: ucx-pr-review
description: Review a UCX pull request diff that an automated review runner supplies, and return findings for the runner to post. Uses the REVIEW.md style and checklist.
metadata:
  resources:
    - REVIEW.md
    - AGENTS.md
---

# UCX PR Review

## Overview

Review the supplied UCX pull request diff and return review findings. The
runner checks out the repository, reads GitHub, collects the PR discussion, and
posts the review. Use only the repository tools that are available to you.

## Sources

The runner supplies `REVIEW.md` and the root `AGENTS.md` as resources. Before
you make a review claim, follow their references to the guides and style docs.

## Review Workflow

1. Use the supplied diff as the source of truth for the proposed changes and
   for line anchors. A file that you read with a repository tool can show the
   code before the change. If a file read and the diff do not agree, use the
   diff.
2. Before you report a finding, examine the related code with the repository
   tools. Examine definitions, callers, state transitions, cleanup paths,
   tests, and build metadata.
3. Use the supplied PR metadata, CI results, and PR discussion as evidence. Do
   not claim test or CI results that the supplied evidence does not show.
4. If the runner does not supply the PR title and description, do not check
   them. If the runner does not supply CI results, do not check CI.
5. Apply the PR size and scope rules from `REVIEW.md`.
6. Apply the risk order and checklist from `REVIEW.md`.
7. Apply the candidate comment gate to each candidate finding.

## Candidate Comment Gate

Keep a finding only if all of these are true:

- The supplied changes introduce or expose the issue, and the latest diff
  still contains it.
- The related definitions, callers, and existing safeguards do not make the
  concern invalid.
- The supplied PR discussion does not already contain the same concern or an
  adequate answer to it. Compare by meaning, not by exact words or line.
- The comment follows the Comment Style section of `REVIEW.md`. Check the
  severity prefix and the length. Use a `suggestion` block when the fix is
  mechanical.

## Output

- Return only actionable findings. For each finding, give the repository path,
  the right-side diff line when one applies, and the exact comment text.
- If there are no findings, return `No findings.`
