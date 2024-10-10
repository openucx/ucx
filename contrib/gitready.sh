#!/bin/bash -eu

git_commit() {
    if ! git diff-index --quiet HEAD; then
        git add -p
        # returns 1 if cached index is not empty
        git diff --cached --exit-code || git commit -m "$title"
        git reset --hard
    fi
}

ucx=https://github.com/openucx/ucx.git
upstream=$(git remote -v  | grep -P "[\t ]${ucx}[\t ].*fetch" | cut -f 1 | head -n 1)
base=$(git merge-base "$upstream"/master HEAD)

if ! git diff-index --quiet HEAD; then
    echo "error: tree not clean"
    exit 1
fi

ok=1
for sha1 in $(git log "$base"..HEAD --format="%h")
do
    title=$(git log -1 --format="%s" "$sha1")
    if [ ${#title} -gt "80" ]
    then
        echo "commit title too long (${#title}): '$title'"
        ok=0
    elif echo "$title" | grep -qP '^Merge |^[0-9A-Z/_\-]*: \w'; then
        echo "good commit title: '$title'"
    else
        echo "bad commit title: '$title'"
        ok=0
    fi
done

if [ "$ok" -eq "0" ]
then
    echo "error: fix commit title"
    exit 1
fi

# Indent
module load dev/llvm-ucx || :
git clang-format --diff "$base" HEAD | patch -p1
module unload dev/llvm-ucx || :

git_commit

# Codespell
TMP_ENV=/tmp/codespell_env
python3 -m venv "$TMP_ENV"
source "$TMP_ENV/bin/activate"
pip3 install codespell
codespell --write-changes || :

git_commit

# Pushing
if [ "${1-}" = "--push" ]
then

    opt="${2-}"
    remote="${opt%%/*}"
    branch="${opt#*/}"

    if [ "$remote" = "$opt" ] || [ -z "$remote" ] || [ -z "$branch" ]
    then
        echo "error: specify push location with '--push <remote>/<branch_name>'"
        exit 1
    fi

    cmd="git push $remote HEAD:refs/heads/$branch"
    echo "$cmd"
    echo "<enter> or <ctrl-c> to abort"
    read -r
    $cmd
fi
