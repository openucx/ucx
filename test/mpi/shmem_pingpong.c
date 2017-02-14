/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <shmem.h>


static inline void wait_until(void *shared_buffer, long sn)
{
    if (0) {
        while (*(volatile long*)shared_buffer != sn);
    } else {
        shmem_wait_until(shared_buffer, SHMEM_CMP_EQ, sn);
    }
}

static void run(void *shared_buffer)
{
    const long iters = 100000;
    struct timeval tv_start, tv_end;
    double latency_usec;
    long sn;
    long i;

    gettimeofday(&tv_start, NULL);

    sn = 1;
    if (shmem_my_pe() == 0) {
        for (i = 0; i < iters; ++i) {
            shmem_long_p(shared_buffer, sn, 1);
            wait_until(shared_buffer, sn);
            ++sn;
        }
    } else {
        for (i = 0; i < iters; ++i) {
            wait_until(shared_buffer, sn);
            shmem_long_p(shared_buffer, sn, 0);
            ++sn;
        }
    }

    shmem_quiet();

    gettimeofday(&tv_end, NULL);

    latency_usec = ((tv_end.tv_sec * 1e6 + tv_end.tv_usec) -
                    (tv_start.tv_sec * 1e6 + tv_start.tv_usec)) / iters / 2;

    if (shmem_my_pe() == 0) {
        printf("%9ld   %.3f usec\n", iters, latency_usec);
    }
}

int main(int argc, char **argv)
{
    int i, iters;
    void *shared_buffer;

    if (argc > 1) {
        iters = atoi(argv[1]);
    } else {
        iters = 4;
    }

    start_pes(0);

    shared_buffer = shmalloc(sizeof(long));

    /* coverity[tainted_data] */
    for (i = 0; i < iters; ++i) {
        memset(shared_buffer, 0, sizeof(long));
        shmem_barrier_all();
        run(shared_buffer);
    }

    shmem_barrier_all();

    shfree(shared_buffer);
    return 0;
}
