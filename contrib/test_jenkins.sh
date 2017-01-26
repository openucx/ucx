#!/bin/bash -eExl

rc=0

WORKSPACE=${WORKSPACE:=$PWD}

if [ -z "$BUILD_NUMBER" ]; then
    echo Running interactive
    BUILD_NUMBER=1
    WS_URL=file://$WORKSPACE
    JENKINS_RUN_TESTS=yes
else
    echo Running under jenkins
    WS_URL=$JOB_URL/ws
fi

make_opt="-j$(($(nproc) / 2 + 1))"
ucx_inst=${WORKSPACE}/install

# indicate to coverity which files to exclude from report
cov_exclude_file_list="external/jemalloc jemalloc"


echo Starting on host: $(hostname)

cd ${WORKSPACE}

echo "Autogen"
./autogen.sh
make $make_opt distclean||:

echo "Making a directory for test build"
rm -rf build-test
mkdir -p build-test
cd build-test

echo "Build docs only"
../configure --with-docs-only
make $make_opt docs

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
        TIMEOUT="timeout 160m"
        TIMEOUT_VALGRIND="timeout 200m"
    else
        AFFINITY=""
        TIMEOUT=""
        TIMEOUT_VALGRIND=""
    fi

    # Load newer doxygen
    module load tools/doxygen-1.8.11

    echo "Build gtest"
    module load hpcx-gcc

    make $make_opt clean

    # todo: check in -devel mode as well
    ../contrib/configure-release --with-mpi --prefix=$ucx_inst
    make $make_opt install

    # Prevent out tests from using installed UCX libraries
    export LD_LIBRARY_PATH=${ucx_inst}/lib:$LD_LIBRARY_PATH

    # check whether installation is valid (it compiles examples at least)
    make installcheck

    ucx_inst_ptest=$ucx_inst/share/ucx/perftest

    # hack for perftest, no way to override params used in batch
    # todo: fix in perftest
    sed -s 's,-n [0-9]*,-n 1000,g' $ucx_inst_ptest/msg_pow2 | sort -R > $ucx_inst_ptest/msg_pow2_short
    cat $ucx_inst_ptest/test_types | sort -R > $ucx_inst_ptest/test_types_short

    opt_perftest_common="-b $ucx_inst_ptest/test_types_short -b $ucx_inst_ptest/msg_pow2_short -w 1"

    # show UCX libraries being used
    ldd $ucx_inst/bin/ucx_perftest
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

    # compile and run UCP hello world example
    gcc -o ./ucp_hello_world ${ucx_inst}/share/ucx/examples/ucp_hello_world.c -lucp -lucs -I${ucx_inst}/include -L${ucx_inst}/lib

	# debug settings
    export UCX_HANDLE_ERRORS=freeze,bt
    export UCX_ERROR_SIGNALS=SIGILL,SIGSEGV,SIGBUS,SIGFPE,SIGPIPE
    export UCX_ERROR_MAIL_TO=$ghprbActualCommitAuthorEmail
    export UCX_ERROR_MAIL_FOOTER=$JOB_URL/$BUILD_NUMBER/console

	# hello-world example
    UCP_TEST_HELLO_WORLD_PORT=$(( 10000 + ${BASHPID} ))
    for test_mode in -w -f -b ; do
        echo Running UCP hello world server with mode ${test_mode}
        ./ucp_hello_world ${test_mode} -p ${UCP_TEST_HELLO_WORLD_PORT} &
        hw_server_pid=$!

        sleep 3

        #need to be ran in background to reflect application PID in $!
        echo Running UCP hello world client with mode ${test_mode}
        ./ucp_hello_world ${test_mode} -n $(hostname) -p ${UCP_TEST_HELLO_WORLD_PORT} &
        hw_client_pid=$!

        # make sure server process in not running
        wait ${hw_server_pid} ${hw_client_pid}
    done
    rm -f ./ucp_hello_world

    # compile and then run UCT example to make sure it's not broken by UCX API changes
    mpicc -o ./active_message $ucx_inst/share/ucx/examples/active_message.c -luct -lucs -I${ucx_inst}/include -L${ucx_inst}/lib

    for dev in $(ibstat -l); do
        hca="${dev}:1"

        if [[ $dev =~ .*mlx5.* ]]; then
            opt_perftest="$opt_perftest_common -b $ucx_inst_ptest/transports"
        else
            opt_perftest="$opt_perftest_common -x rc"
        fi

        echo Running ucx_perf kit on $hca
        mpirun -np 2 -x UCX_ERROR_SIGNALS -x UCX_HANDLE_ERRORS -mca pml ob1 -mca btl sm,self $AFFINITY $ucx_inst/bin/ucx_perftest -d $hca $opt_perftest

        echo Running active_message example on $hca with rc
         mpirun -np 2 -x UCX_ERROR_SIGNALS -x UCX_HANDLE_ERRORS -mca pml ob1 -mca btl sm,self -mca coll ^hcoll -x UCX_IB_ETH_PAUSE_ON=y ./active_message $hca "rc"

        # todo: add csv generation

    done

    for mm_device in sysv posix; do
        echo Running ucx_perf kit with shared memory
        mpirun -np 2 -x UCX_ERROR_SIGNALS -x UCX_HANDLE_ERRORS -mca pml ob1 -mca btl sm,self $AFFINITY $ucx_inst/bin/ucx_perftest -d $mm_device $opt_perftest_common -x mm
    done

    rm -f ./active_message

    for tname in malloc_hooks external_events flag_no_install; do
        echo "Running memory hook (${tname}) on MPI"
        mpirun -np 1 -x UCX_ERROR_SIGNALS -x UCX_HANDLE_ERRORS -mca pml ob1 -mca btl sm,self -mca coll ^hcoll,ml $AFFINITY ./test/mpi/test_memhooks -t $tname
    done

    echo "Running memory hook (malloc_hooks) on MPI with LD_PRELOAD"
    ucm_lib=$PWD/src/ucm/.libs/libucm.so
    ls -l $ucm_lib
    mpirun -np 1 -x UCX_ERROR_SIGNALS -x UCX_HANDLE_ERRORS -mca pml ob1 -mca btl sm,self -mca coll ^hcoll,ml -x LD_PRELOAD=$ucm_lib $AFFINITY ./test/mpi/test_memhooks -t malloc_hooks


    echo "Check --disable-numa compilation option"
    ../contrib/configure-devel --prefix=$ucx_inst --disable-numa
    make $make_opt clean
    make $make_opt all
    make $make_opt distclean

    module unload hpcx-gcc

    module load intel/ics
    ../contrib/configure-devel --prefix=$ucx_inst CC=icc
    make $make_opt clean
    make $make_opt all
    make $make_opt distclean
    module unload intel/ics

    ../contrib/configure-devel --prefix=$ucx_inst
    make $make_opt all

    echo "Running ucx_info"
    $AFFINITY $TIMEOUT ./src/tools/info/ucx_info -f -c -v -y -d -b -p -w -e -uart

    echo "Running profiling test"
    UCX_PROFILE_MODE=log UCX_PROFILE_FILE=ucx_jenkins.prof ./test/apps/test_profiling
    ${ucx_inst}/bin/ucx_read_profile -r ucx_jenkins.prof | grep "printf" -C 20
    ${ucx_inst}/bin/ucx_read_profile -r ucx_jenkins.prof | grep -q "calc_pi"
    ${ucx_inst}/bin/ucx_read_profile -r ucx_jenkins.prof | grep -q "print_pi"

    echo "Running dlopen test"
    strace ./test/apps/test_profiling &> strace.log
    ! grep '^socket' strace.log

    export GTEST_RANDOM_SEED=0
    export GTEST_SHUFFLE=1
    export GTEST_TAP=2

    export GTEST_REPORT_DIR=$WORKSPACE/reports/tap
    mkdir -p $GTEST_REPORT_DIR

    echo "Running unit tests"
    $AFFINITY $TIMEOUT make -C test/gtest test
    (cd test/gtest && rename .tap _gtest.tap *.tap && mv *.tap $GTEST_REPORT_DIR)

    echo "Running valgrind tests"
    if [ $(valgrind --version) != "valgrind-3.10.0" ]
    then
        module load tools/valgrind-latest
    fi
    $AFFINITY $TIMEOUT_VALGRIND make -C test/gtest VALGRIND_EXTRA_ARGS="--xml=yes --xml-file=valgrind.xml --child-silent-after-fork=yes" test_valgrind
    (cd test/gtest && rename .tap _vg.tap *.tap && mv *.tap $GTEST_REPORT_DIR)
    module unload tools/valgrind-latest

    echo "Build with coverity"
    module load tools/cov
    cov_build_id="cov_build_${BUILD_NUMBER}"
    cov_build="$WORKSPACE/$cov_build_id"
    rm -rf $cov_build
    make clean
    cov-build --dir $cov_build make $make_opt all

    set +eE
    for excl in $cov_exclude_file_list; do
        cov-manage-emit --dir $cov_build --tu-pattern "file('$excl')" delete
    done
    set -eE

    cov-analyze --dir $cov_build
    nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
    rc=$(($rc+$nerrors))

    index_html=$(cd $cov_build && find . -name index.html | cut -c 3-)
    cov_url="$WS_URL/$cov_build_id/${index_html}"
    rm -f jenkins_sidelinks.txt
    echo 1..1 > coverity.tap
    if [ $nerrors -gt 0 ]; then
        cov-format-errors --dir $cov_build --emacs-style
        echo "not ok 1 Coverity Detected $nerrors failures # $cov_url" >> coverity.tap
    else
        echo "ok 1 Coverity found no issues" >> coverity.tap
    fi
    echo Coverity report: $cov_url
    printf "%s\t%s\n" Coverity $cov_url >> jenkins_sidelinks.txt
    module unload tools/cov
fi


exit $rc
