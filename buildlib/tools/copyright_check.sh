#!/bin/bash -x

WORKSPACE=$SYSTEM_DEFAULTWORKINGDIRECTORY
IMAGE="rdmz-harbor.rdmz.labs.mlnx/toolbox/header_check"
FILES=$(git diff --name-only HEAD^ HEAD | paste -sd ",")

HEADER_CHECK_TOOL="docker run --rm \
    --user $(id -u):$(id -g) \
    -v $PWD:$PWD -w $PWD \
    $IMAGE"

$HEADER_CHECK_TOOL \
    --config "${WORKSPACE}"/buildlib/tools/copyright-check-map.yaml \
    --headers-types "${WORKSPACE}"/buildlib/tools/copyright-header-types.json \
    --path "$FILES" \
    --git-repo "$WORKSPACE" | tee copyrights.log

exit_code=$?
echo "exit_code=${exit_code}"
# Correct error code is not returned by the script, need to check output file if its empty it failed
if [[ ! -s copyrights.log ]]; then
    echo "copyrights.log is empty which means the script failed internally"
    exit 1
fi
set +eE
grep -rn ERROR copyrights.log
exit_code=$?
set -eE
if [ ${exit_code} -eq 0 ]; then
    echo "Please refer to https://confluence.nvidia.com/pages/viewpage.action?pageId=788418816"
    ${HEADER_CHECK_TOOL} \
      --config "${WORKSPACE}"/buildlib/tools/copyright-check-map.yaml \
      --headers-types "${WORKSPACE}"/buildlib/tools/copyright-header-types.json \
      --path "$FILES" \
      --repair \
      --git-repo "${WORKSPACE}" | tee copyrights_repair.log
    # create list of modified files
    # needed for new git versions (from the check header docker image)
    git config --global --add safe.directory "$WORKSPACE"
    # files=$(git status | grep 'modified:' | awk '{print $NF}'  )
    mkdir "$WORKSPACE"/repaired_files/
    git status | grep 'modified:' | awk '{print $NF}' | xargs -I{} cp --parents {} "$WORKSPACE"/repaired_files/
    # cp --parents "$files" "$WORKSPACE"/repaired_files/
    
    cd "$WORKSPACE"/repaired_files/
    tar -czf "$WORKSPACE"/copyright_repaired_files_"$BUILD_BUILDID".tar.gz .
    exit 1
fi
