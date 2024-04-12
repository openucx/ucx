#!/bin/bash -eEx

source buildlib/tools/common.sh

prepare_build
clang --version
gcc --version
cppcheck --version
${WORKSPACE}/contrib/configure-release

report_dir=$(readlink -f ${WORKDIR}/static_check)
mkdir -p $report_dir
echo "##vso[task.setvariable variable=reportExists]True"
echo "##vso[task.setvariable variable=report_dir]$report_dir"

export PATH="`csclng --print-path-to-wrap`:`cscppc --print-path-to-wrap`:`cswrap --print-path-to-wrap`:$PATH"
export CSCLNG_ADD_OPTS="-Xanalyzer:-analyzer-output=html:-o:$report_dir"
set -o pipefail
make -j`nproc` |& tee $report_dir/compile.log
set +o pipefail

cs_errors="cs.err"
cslinker --quiet $report_dir/compile.log \
    | csgrep --mode=json --path ${WORKSPACE} --strip-path-prefix ${WORKSPACE} \
    | csgrep --mode=json --invert-match --path 'conftest.c' \
    | csgrep --mode=grep --invert-match --event "internal warning" --prune-events=1 \
    > $cs_errors

if [ -s $cs_errors ]; then
    echo "static checkers found errors:"
    cat $cs_errors
    echo "##vso[task.logissue type=error]static checkers found errors"
    echo "##vso[task.complete result=Failed;]"
else
    echo "No errors reported by static checkers"
fi

