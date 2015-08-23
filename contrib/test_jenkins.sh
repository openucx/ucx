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
ucx_inst=${WORKSPACE}/install

echo Starting on host: $(hostname)

echo "Autogen"
./autogen.sh
make $make_opt distclean||:

echo "Making a directory for test build"
rm -rf build-test
mkdir -p build-test
cd build-test

echo "Build release"
../contrib/configure-release
make $make_opt
make $make_opt distcheck

if [ -x /usr/bin/dpkg-buildpackage ]; then
    echo "Build on debian"
    dpkg-buildpackage -us -uc
else
    echo "Build rpms"
    ../contrib/buildrpm.sh -s -b
fi

echo "Build docs"
make $make_opt docs

echo "Running ucx_info"
./src/tools/info/ucx_info -v -f
./src/tools/info/ucx_info -c

echo "Build without IB verbs"
../contrib/configure-release --without-verbs
make $make_opt

if [ -n "$JENKINS_RUN_TESTS" ]; then
    # Set CPU affinity to 2 cores, for performance tests
    if [ -n "$EXECUTOR_NUMBER" ]; then
        AFFINITY="taskset -c $(( 2 * EXECUTOR_NUMBER ))","$(( 2 * EXECUTOR_NUMBER + 1))"
        TIMEOUT="timeout 80m"
    else
        AFFINITY=""
        TIMEOUT=""
    fi

    echo "Build gtest"
    module load hpcx-gcc
    make $make_opt clean

    # todo: check in -devel mode as well
    ../contrib/configure-release --with-mpi --prefix=$ucx_inst
    make $make_opt install

    ucx_inst_ptest=$ucx_inst/share/ucx/perftest

    # hack for perftest, no way to override params used in batch
    # todo: fix in perftest


    sed -s 's,-n [0-9]*,-n 1000,g' $ucx_inst_ptest/msg_pow2 | sort -R > $ucx_inst_ptest/msg_pow2_short
    cat $ucx_inst_ptest/test_types | sort -R > $ucx_inst_ptest/test_types_short

    opt_perftest_common="-b $ucx_inst_ptest/test_types_short -b $ucx_inst_ptest/msg_pow2_short -w 1"

    for dev in $(ibstat -l); do
        hca="${dev}:1"

        if [[ $dev =~ .*mlx5.* ]]; then
            opt_perftest="$opt_perftest_common -b $ucx_inst_ptest/transports"
        else
            opt_perftest="$opt_perftest_common -x rc"
        fi

        echo Running ucx_perf kit on $hca
        mpirun -np 2 $AFFINITY $ucx_inst/bin/ucx_perftest -d $hca $opt_perftest

        # todo: add csv generation

    done

    ../contrib/configure-devel --with-mpi --prefix=$ucx_inst
    make $make_opt all

    module unload hpcx-gcc


    echo "Running ucx_info"
    $AFFINITY $TIMEOUT ./src/tools/info/ucx_info -f -c -v -y -d -b

    echo "Running unit tests"
    $AFFINITY $TIMEOUT make -C test/gtest test UCS_HANDLE_ERRORS=bt

    echo "Running valgrind tests"
    module load tools/valgrind-latest
    $AFFINITY $TIMEOUT make -C test/gtest UCS_HANDLE_ERRORS=bt VALGRIND_EXTRA_ARGS="--xml=yes --xml-file=valgrind.xml" test_valgrind
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

    index_html=$(cd $cov_build && find . -name index.html | cut -c 3-)
    cov_url="$WS_URL/$cov_build_id/${index_html}"
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


exit $rc
