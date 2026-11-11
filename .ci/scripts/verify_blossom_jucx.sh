#!/bin/bash
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
set -euo pipefail

revision=b3e859d0f7741c3cde87c7d838744c9bc95d529c
test "$(uname -m)" = aarch64
test "$(getconf PAGESIZE)" = 65536
printf 'JUCX host=%s page_size=%s PR=12311 SHA=%s\n' \
    "$(hostname)" "$(getconf PAGESIZE)" "$revision"
ulimit -l unlimited
export UCX_HANDLE_ERRORS=bt UCX_LOG_LEVEL=info UCX_TLS=rc
unset UCX_NET_DEVICES
if [ ! -d regression-jucx/.git ]; then
    git clone --no-checkout https://github.com/openucx/ucx.git regression-jucx
fi
cd regression-jucx
git fetch --no-tags origin "$revision"
git checkout --detach "$revision"
test "$(git rev-parse HEAD)" = "$revision"
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64
export PATH=$JAVA_HOME/bin:$PATH
if [ ! -f Makefile ]; then
    ./autogen.sh
    ./contrib/configure-devel --prefix="$PWD/install" --with-java \
        --enable-gtest=no --with-cuda=no
fi
make -j24
test -f src/uct/ib/.libs/libuct_ib.so
test -f src/uct/ib/rdmacm/.libs/libuct_rdmacm.so
test -n "$(ls /sys/class/infiniband/)"
mkdir -p bindings/java/resources/ucx
cp src/uct/ib/.libs/libuct_ib.so \
    src/uct/ib/rdmacm/.libs/libuct_rdmacm.so bindings/java/resources/ucx/
for jdk in java-8-openjdk-arm64 java-11-openjdk-arm64; do
    export JAVA_HOME=/usr/lib/jvm/$jdk
    export PATH=$JAVA_HOME/bin:$PATH
    printf 'VERIFY PR=12311 SHA=%s JDK=%s\n' "$revision" "$jdk"
    rm -f bindings/java/src/main/native/build-java/surefire-reports/TEST-*.xml
    for selection in UcpMemoryTest#testMmapFile UcpListenerTest#testConnectionHandler; do
        timeout 300 make -C bindings/java/src/main/native test \
            MVN="mvn -Dtest=$selection -DfailIfNoTests=true"
    done
    test ! -d bindings/java/resources/ucx
    test ! -d bindings/java/src/main/native/build-java/test-classes/ucx
    python3 - <<'PY'
import glob
import xml.etree.ElementTree as ET

reports = glob.glob('bindings/java/src/main/native/build-java/surefire-reports/TEST-*.xml')
cases = [case for path in reports for case in ET.parse(path).getroot().iter('testcase')]
assert {c.attrib['name'] for c in cases} == {'testMmapFile', 'testConnectionHandler'}
assert len(cases) == 2, 'Expected exactly two selected JUCX tests'
assert not any(c.find(tag) is not None for c in cases
               for tag in ('failure', 'error', 'skipped')), 'Test failed or skipped'
print('VERIFIED PR=12311 tests=2 failures=0 skipped=0')
PY
done
