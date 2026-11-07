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
set -euo pipefail

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
    tr -d '[:blank:]\r' <<< "${1,,}" | sed -e 's/#.*//' -e 's/^@//' -e '/^$/d'
}

# Fail loudly on an empty or malformed list, not as missing CLAs
check_list() {
    local bad
    if [ -z "$2" ]; then
        echo "::error::$1 is empty"
        exit 1
    fi
    bad=$({ grep -nvE "$3" <<< "$2" || true; } | cut -d: -f1 | paste -sd, -)
    if [ -n "$bad" ]; then
        echo "::error::$1 has malformed entries at positions $bad" \
             "(blank and comment lines not counted)"
        exit 1
    fi
}

emails=$(read_list "${CLA_INDIVIDUALS:-}")
domains=$(read_list "${CLA_ENTITIES:-}")
check_list CLA_INDIVIDUALS "$emails" '^[^@]+@[a-z0-9.-]+$'
check_list CLA_ENTITIES "$domains" '^[a-z0-9-]+(\.[a-z0-9-]+)+$'

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Commits only, no trees or blobs: PR code is never checked out
git init -q "$work/repo"
git -C "$work/repo" fetch -q --filter=tree:0 --no-tags \
    "https://github.com/$repo.git" \
    "+refs/heads/master:refs/remotes/origin/master" \
    "+refs/heads/v*:refs/remotes/origin/v*" "$head_sha"
mapfile -t new < <(git -C "$work/repo" rev-list "$head_sha" \
    --not origin/master --remotes='origin/v*.*.x')
echo "Checking ${#new[@]} new commits"

# GitHub's web-flow committer signs web UI commits and sets their author
# from the acting account, so the author email stands for both. Bots get
# no exemption: any fork can mint a signed bot-authored commit. Unsigned
# commits report their emails too, so one run shows everything to fix.
declare -A missing=()
unsigned=()
for sha in "${new[@]}"; do
    gh api "repos/$repo/commits/$sha" --jq '
        (select(.commit.verification.verified | not) |
         "unsigned \(.commit.verification.reason)"),
        "author \(.commit.author.email)",
        (select(.committer.login != "web-flow" and
                .commit.committer.email != .commit.author.email) |
         "committer \(.commit.committer.email)")' > "$work/commit.txt"

    short=${sha::10}
    while read -r kind rest; do
        if [ "$kind" = "unsigned" ]; then
            unsigned+=("$short ($rest)")
            continue
        fi
        email=${rest,,}
        email=${email:-"(no email)"}
        if grep -qxF -- "$email" <<< "$emails" ||
           { [[ $email == *@* ]] &&
             grep -qxF -- "${email##*@}" <<< "$domains"; }; then
            continue
        fi
        missing["$kind $email"]+=" $short"
    done < "$work/commit.txt"
done

# Update the PR comment left by an earlier run; create one on failure
post_comment() {
    local id
    if [ -n "$dry_run" ]; then
        echo "[dry-run] would post the PR comment"
        return
    fi
    id=$(gh api "repos/$repo/issues/$pr/comments" --paginate --jq "first(.[] |
        select(.user.type == \"Bot\" and (.body | startswith(\"$MARKER\"))) |
        .id)") || return 1
    id=${id%%$'\n'*}
    if [ -n "$id" ]; then
        gh api -X PATCH "repos/$repo/issues/comments/$id" \
            -F body=@"$work/summary.md" > /dev/null
    elif [ "$failed" = 1 ]; then
        gh api "repos/$repo/issues/$pr/comments" \
            -F body=@"$work/summary.md" > /dev/null
    fi
}

failed=$(( ${#missing[@]} + ${#unsigned[@]} > 0 ))
{
    echo "$MARKER"
    if [ "$failed" = 0 ]; then
        echo "### UCF CLA check passed"
        echo "All commits are covered by a signed UCF CLA."
    else
        echo "### UCF CLA check failed"
    fi
    if [ ${#unsigned[@]} -gt 0 ]; then
        printf -v list '%s, ' "${unsigned[@]}"
        echo "- Commits without a verified signature: ${list%, }."
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        echo "- No signed CLA on file for these commit emails:"
        while IFS= read -r key; do
            echo "  - \`$key\`:${missing[$key]}"
        done < <(printf '%s\n' "${!missing[@]}" | sort)
    fi
    if [ "$failed" = 1 ]; then
        echo
        echo "See items 1 and 2 of [CONTRIBUTING.md]($CONTRIBUTING) for how"
        echo "to sign the CLA and your commits. Once your email is on the CLA"
        echo "list, push a new commit or ask for a re-run of this check."
    fi
} > "$work/summary.md"
tail -n +2 "$work/summary.md" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
post_comment || echo "::warning::Could not post the PR comment"
if [ "$failed" = 1 ]; then
    echo "::error::UCF CLA check failed, see the job summary"
    exit 1
fi
