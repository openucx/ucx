#!/usr/bin/env bash
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Check that every commit a PR adds is covered by a UCF CLA: it must have a
# verified signature, and its author and committer emails must be listed as
# individual signers or belong to a corporate signer's email domain.
# Commits already on master or a release branch were checked when they landed.
# Findings are written to the job summary and posted as one PR comment,
# updated on every run.
#
# Usage: cla_check.sh --repo <owner/name> --pr <number> --head-sha <sha>
#        [--dry-run]
# Environment: CLA_INDIVIDUALS (signer emails), CLA_ENTITIES (signer domains),
#              one entry per line, # starts a comment; GH_TOKEN for gh.
# --dry-run skips posting the PR comment.
set -eEuo pipefail

usage() {
    sed -n '/^# Usage:/,/^# --dry-run/s/^# //p' "$0"
    exit 2
}

MARKER='<!-- ucf-cla-check -->'
CONTRIBUTING=https://github.com/openucx/ucx/blob/master/.github/CONTRIBUTING.md
repo="" pr="" head_sha="" dry_run=""
while [ $# -gt 0 ]; do
    case $1 in
    --repo)     repo=$2; shift 2 ;;
    --pr)       pr=$2; shift 2 ;;
    --head-sha) head_sha=$2; shift 2 ;;
    --dry-run)  dry_run=1; shift ;;
    *)          usage ;;
    esac
done
[ -n "$repo" ] && [ -n "$pr" ] && [ -n "$head_sha" ] || usage

read_list() {
    tr -d '[:blank:]\r' <<< "$1" | sed -e 's/#.*//' -e 's/^@//' |
        tr '[:upper:]' '[:lower:]' | grep -v '^$' || true
}

emails=$(read_list "${CLA_INDIVIDUALS:-}")
domains=$(read_list "${CLA_ENTITIES:-}")
# Fail loudly on an empty or malformed list, not as missing CLAs
if [ -z "$emails" ] || [ -z "$domains" ]; then
    echo "::error::A CLA list secret is empty"
    exit 1
fi
if grep -qvE '^[^@]+@[a-z0-9.-]+$' <<< "$emails" ||
   grep -qvE '^[a-z0-9.-]+$' <<< "$domains"; then
    echo "::error::A CLA list secret has a malformed entry"
    exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Commits only, no trees or blobs: PR code is never checked out
git init -q "$work/repo"
git -C "$work/repo" fetch -q --filter=tree:0 --no-tags \
    "https://github.com/$repo.git" \
    "+refs/heads/master:refs/remotes/origin/master" \
    "+refs/heads/v*:refs/remotes/origin/v*" "$head_sha"
new=$(git -C "$work/repo" rev-list "$head_sha" --not origin/master \
    --remotes='origin/v*.*.x')
echo "Checking $(wc -w <<< "$new" | tr -d " ") new commits"

# GitHub's web-flow committer signs web UI commits, whose author email
# GitHub takes from the user's verified emails. Bot author and committer
# emails are exempt; unsigned commits report their emails too, so one run
# shows everything to fix.
declare -A missing=()
unsigned=()
for sha in $new; do
    gh api "repos/$repo/commits/$sha" --jq '
        (if .commit.verification.verified then empty
         else "unsigned \(.sha[0:10]) \(.commit.verification.reason)" end),
        (if .author.type == "Bot" then empty
         else "author \(.sha[0:10]) \(.commit.author.email)" end),
        (if .committer.type == "Bot" or .committer.login == "web-flow" or
            .commit.committer.email == .commit.author.email then empty
         else "committer \(.sha[0:10]) \(.commit.committer.email)" end)
        ' > "$work/commit.txt"

    while read -r kind short rest; do
        if [ "$kind" = "unsigned" ]; then
            unsigned+=("$short ($rest)")
            continue
        fi
        email=${rest,,}
        email=${email:-"(no email)"}
        domain=${email##*@}
        if grep -qxF -- "$email" <<< "$emails" ||
           { [ "$domain" != "$email" ] &&
             grep -qxF -- "$domain" <<< "$domains"; }; then
            continue
        fi
        missing["$kind $email"]+=" $short"
    done < "$work/commit.txt"
done

# Update the PR comment left by an earlier run; create one on failure
post_comment() {
    local id
    if [ -n "$dry_run" ]; then
        echo "[dry-run] would $1 the PR comment"
        return
    fi
    id=$(gh api "repos/$repo/issues/$pr/comments" --paginate --jq "first(.[] |
        select(.user.type == \"Bot\" and (.body | startswith(\"$MARKER\"))) |
        .id)") || return 1
    id=${id%%$'\n'*}
    if [ -n "$id" ]; then
        gh api -X PATCH "repos/$repo/issues/comments/$id" \
            -F body=@"$work/summary.md" > /dev/null
    elif [ "$1" = create ]; then
        gh api "repos/$repo/issues/$pr/comments" \
            -F body=@"$work/summary.md" > /dev/null
    fi
}

report() {
    tail -n +2 "$work/summary.md" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
}

if [ ${#missing[@]} -eq 0 ] && [ ${#unsigned[@]} -eq 0 ]; then
    {
        echo "$MARKER"
        echo "### UCF CLA check passed"
        echo "All commits are covered by a signed UCF CLA."
    } > "$work/summary.md"
    report
    post_comment update || echo "::warning::Could not update the PR comment"
    exit 0
fi

{
    echo "$MARKER"
    echo "### UCF CLA check failed"
    if [ ${#unsigned[@]} -gt 0 ]; then
        printf -v list '%s, ' "${unsigned[@]}"
        echo "- Commits without a verified signature: ${list%, }."
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        echo "- No signed CLA on file for these commit emails:"
        while IFS= read -r key; do
            echo "  - $key:${missing[$key]}"
        done < <(printf '%s\n' "${!missing[@]}" | sort)
    fi
    echo
    echo "See items 1 and 2 of [CONTRIBUTING.md]($CONTRIBUTING) for how to sign"
    echo "the CLA and your commits. Once your email is on the CLA list, push a"
    echo "new commit or ask for a re-run of this check."
} > "$work/summary.md"
report
post_comment create || echo "::warning::Could not post the PR comment"
echo "::error::UCF CLA check failed, see the job summary"
exit 1
