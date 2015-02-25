#!/bin/bash -eExl

rc=0

if [ -z "$BUILD_NUMBER" ]; then
    echo Running interactive
    WORKSPACE=$PWD
    BUILD_NUMBER=1
    WS_URL=file://$WORKSPACE
    JENKINS_RUN_TESTS=yes
else
    echo Running under jenkins
    WS_URL=$JOB_URL/ws
fi

make_opt="-j$(($(nproc) - 1))"

echo Starting on host: $(hostname)

echo "Autogen"
./autogen.sh
make distclean||:

echo "Making a directory for test build"
rm -rf build-test && mkdir -p build-test && cd build-test

echo "Build release"
../contrib/configure-release && make $make_opt && make $make_opt distcheck && make docs

echo "Running ucx_info"
./src/tools/info/ucx_info -v -f
./src/tools/info/ucx_info -c

echo "Build without IB verbs"
../contrib/configure-release --without-verbs && make $make_opt

if [ -n "$JENKINS_RUN_TESTS" ]; then
    # Set CPU affinity to 2 cores, for performance tests
    if [ -n "$EXECUTOR_NUMBER" ]; then
        AFFINITY="taskset -c $(( 2 * EXECUTOR_NUMBER ))","$(( 2 * EXECUTOR_NUMBER + 1))"
        TIMEOUT="timeout 10m"
    else
        AFFINITY=""
        TIMEOUT=""
    fi

    echo "Build gtest"
    module load hpcx-gcc
    make clean && ../contrib/configure-devel --with-mpi && make $make_opt
    module unload hpcx-gcc

    echo "Running ucx_info"
    $AFFINITY $TIMEOUT ./src/tools/info/ucx_info -f -c -v -y -d -b

        echo "Running unit tests"
    $AFFINITY $TIMEOUT make -C test/gtest test

    echo "Running valgrind tests"
    module load tools/valgrind-latest
    $AFFINITY $TIMEOUT make -C test/gtest VALGRIND_EXTRA_ARGS="--xml=yes --xml-file=valgrind.xml" test_valgrind
    module unload tools/valgrind-latest

    echo "Build with coverity"
    module load tools/cov
    cov_build_id="cov_build_${BUILD_NUMBER}"
    cov_build="$WORKSPACE/$cov_build_id"
    rm -rf $cov_build
    make clean
    cov-build --dir $cov_build make $make_opt all
    cov-analyze --dir $cov_build
    nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
    rc=$(($rc+$nerrors))

    cov_url="$WS_URL/$cov_build_id/output/errors/index.html"
    rm -f jenkins_sidelinks.txt
    echo 1..1 > coverity.tap
    if [ $nerrors -gt 0 ]; then
        echo "not ok 1 Coverity Detected $nerrors failures # $cov_url" >> coverity.tap
    else
        echo ok 1 Coverity found no issues >> coverity.tap
    fi
    echo Coverity report: $cov_url
    printf "%s\t%s\n" Coverity $cov_url >> jenkins_sidelinks.txt
    module unload tools/cov
fi

#rpm_topdir=$WORKSPACE/rpm-dist
#(make distcheck && rm -rf $rpm_topdir && mkdir -p $rpm_topdir && cd $rpm_topdir && mkdir -p BUILD RPMS SOURCES SPECS SRPMS)
#
#if [ -x /usr/bin/dpkg-buildpackage ]; then
#    echo "Build on debian"
#    dpkg-buildpackage -us -uc
#else
#    echo "Build rpms"
#    rpmbuild -bs --define '_sourcedir .' --define "_topdir $rpm_topdir" --nodeps mxm.spec
#    fn=$(find $rpm_topdir -name "*.src.rpm" -print0)
#    rpmbuild --define "_topdir $rpm_topdir" --rebuild $fn
#fi

exit $rc
