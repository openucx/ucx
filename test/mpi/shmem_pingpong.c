/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <getopt.h>
#include <limits.h>
#include <shmem.h>


#define GLOBAL_DATA_SIZE   1024


static int show_result(const struct timeval *tv_prev,
                       const struct timeval *tv_curr,
                       long iters, size_t msg_size, int force)
{
    double elapsed;

    elapsed = (tv_curr->tv_sec + tv_curr->tv_usec * 1e-6) -
              (tv_prev->tv_sec + tv_prev->tv_usec * 1e-6);

    if (((elapsed >= 1.0) || force) && (shmem_my_pe() == 0)) {
        printf("%ld iterations, %lu bytes, latency: %.3f usec\n",
               iters, msg_size, elapsed * 1e6 / iters / 2.0);
        return 1;
    }

    return 0;
}

static void run_pingpong(char *mem, size_t msg_size, long num_iters, int use_wait,
                         int do_quiet, int use_flag)
{
    struct timeval tv_prev, tv_curr;
    int my_pe, dst_pe;
    volatile int *rsn;
    char *msg;
    int *ssn;
    int sn;
    long i, prev_i;

    msg = malloc(msg_size);
    if (msg == NULL) {
        return;
    }

    memset(msg, 0, msg_size);

    gettimeofday(&tv_prev, NULL);
    prev_i = 0;
    my_pe  = shmem_my_pe();
    dst_pe = 1 - my_pe;
    rsn    = (int*)&mem[msg_size - sizeof(int)];
    ssn    = (int*)&msg[msg_size - sizeof(int)];

    for (i = 0; i < num_iters; ++i) {
        sn   = i & 127;
        *ssn = sn;
        if (my_pe == 0) {
            shmem_putmem(mem, msg, msg_size, dst_pe);
            if (do_quiet) {
                shmem_quiet();
            }
        }
        if (use_wait) {
            shmem_int_wait_until(rsn, SHMEM_CMP_EQ, sn);
        } else {
            while (*rsn != sn);
        }
        if (my_pe == 1) {
            if (use_flag) {
                shmem_putmem(mem, msg, msg_size - sizeof(int), dst_pe);
                shmem_fence();
                shmem_int_put((int*)rsn, ssn, 1, dst_pe);
            } else {
                shmem_putmem(mem, msg, msg_size, dst_pe);
            }
            if (do_quiet) {
                shmem_quiet();
            }
        }
        if ((i % 1000) == 0) {
            gettimeofday(&tv_curr, NULL);
            if (show_result(&tv_prev, &tv_curr, i - prev_i, msg_size, 0)) {
                prev_i = i;
                tv_prev = tv_curr;
            }
        }
    }

    gettimeofday(&tv_curr, NULL);
    show_result(&tv_prev, &tv_curr, num_iters - prev_i, msg_size, 1);
    free(msg);
}

static void usage()
{
    printf("Usage:   shmem_pingpong [options]\n");
    printf("\n");
    printf("Options are:\n");
    printf("  -n <iters>     Specify number of iterations to run (default: 10000).\n");
    printf("  -s <size>      Specify message size (default: 4 bytes).\n");
    printf("  -w             Wait for data using shmem_wait_until() (default: poll on memory).\n");
    printf("  -f             Send data and flag separately with shmem_fence() in-between.\n");
    printf("  -g             Use global data (default: heap).\n");
    printf("  -q             call shmem_quiet() after every shmem_put().\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    static char global_buffer[GLOBAL_DATA_SIZE];
    int use_wait, use_global, do_quiet, use_flag;
    size_t msg_size;
    long num_iters;
    int my_pe;
    char *mem;
    int c;

    start_pes(0);

    my_pe = shmem_my_pe();

    if (shmem_n_pes() != 2) {
        fprintf(stderr, "This test requires exactly 2 processes\n");
        return -1;
    }

    num_iters  = 10000;
    use_global = 0;
    use_wait   = 0;
    do_quiet   = 0;
    use_flag   = 0;
    msg_size   = 8;
    while ((c = getopt (argc, argv, "n:s:wgqfh")) != -1) {
        switch (c) {
            break;
        case 'n':
            num_iters = atol(optarg);
            if (num_iters == 0) {
                num_iters = LONG_MAX;
            }
            break;
        case 'w':
            use_wait = 1;
            break;
        case 'g':
            use_global = 1;
            break;
        case 'q':
            do_quiet = 1;
            break;
        case 'f':
            use_flag = 1;
            break;
        case 's':
            msg_size = atol(optarg);
            break;
        case 'h':
        default:
            if (my_pe == 0) {
                usage();
            }
            return 0;
        }
    }

    if (msg_size < sizeof(int)) {
        fprintf(stderr, "message size must be at least %lu\n", sizeof(int));
        return -1;
    }

    if (use_global) {
        if (msg_size <= GLOBAL_DATA_SIZE) {
            mem = global_buffer;
        } else {
            fprintf(stderr, "global data can be used only up to %lu bytes\n",
                    (size_t)GLOBAL_DATA_SIZE);
            return -1;
        }
    } else {
        mem = shmalloc(msg_size);
    }

    memset(mem, 0xff, msg_size);

    shmem_barrier_all();

    run_pingpong(mem, msg_size, num_iters, use_wait, do_quiet, use_flag);

    shmem_barrier_all();

    if (!use_global) {
        shfree(mem);
    }

    shmem_finalize();
    return 0;
}
