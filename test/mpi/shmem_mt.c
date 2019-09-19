/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#define _GNU_SOURCE // for pthread_setaffinity_np
#include <shmem.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


#define _printf(_fmt, ...) \
    { \
        if (shmem_my_pe() == 0) { \
            printf(_fmt "\n", ## __VA_ARGS__); \
        } \
    }

typedef enum {
    TEST_NONE = 0,
    TEST_PUT,
    TEST_GET,
    TEST_GET_NBI,
    TEST_LAST
} test_op_t;

typedef enum {
    SHMEM_CTX_TYPE_NONE,   /* don't use shmem contexts */
    SHMEM_CTX_TYPE_SHARED,
    SHMEM_CTX_TYPE_PRIVATE,
} shmem_ctx_type_t;

struct {
    /* parameters */
    int                num_threads;
    long               iters;
    int                nb_window;
    test_op_t          operation;
    shmem_ctx_type_t   shmem_ctx_type;
    size_t             message_size;
    size_t             offset; /* distance between dest ptr of threads */

    /* global state */
    void               *buffer;
    pthread_barrier_t  barrier;
    int                dst_pe;

} test_ctx = {
   .num_threads      = 1,
   .iters            = 2000000,
   .nb_window        = 64,
   .operation        = TEST_NONE,
   .shmem_ctx_type   = SHMEM_CTX_TYPE_NONE,
   .message_size     = 8,
   .offset           = 128
};

typedef struct {
    pthread_t        pthread;
    int              thread_num;
    shmem_ctx_t      shmem_ctx;
    void             *local;
    void             *remote;
} thread_ctx_t;


typedef void (*test_func_t)(thread_ctx_t *thread_ctx);

static void do_put(thread_ctx_t *thread_ctx)
{
    long i;

    if (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_NONE) {
        for (i = 0; i < test_ctx.iters; ++i) {
            shmem_putmem(thread_ctx->remote, thread_ctx->local,
                         test_ctx.message_size, test_ctx.dst_pe);
        }
    } else {
        for (i = 0; i < test_ctx.iters; ++i) {
            shmem_ctx_putmem(thread_ctx->shmem_ctx,
                             thread_ctx->remote, thread_ctx->local,
                             test_ctx.message_size, test_ctx.dst_pe);
        }
    }
}

static void do_get(thread_ctx_t *thread_ctx)
{
    long i;

    if (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_NONE) {
        for (i = 0; i < test_ctx.iters; ++i) {
            shmem_getmem(thread_ctx->local, thread_ctx->remote,
                         test_ctx.message_size, test_ctx.dst_pe);
        }
    } else {
        for (i = 0; i < test_ctx.iters; ++i) {
            shmem_ctx_getmem(thread_ctx->shmem_ctx,
                             thread_ctx->local, thread_ctx->remote,
                             test_ctx.message_size, test_ctx.dst_pe);
        }
    }
}

static void do_get_nbi(thread_ctx_t *thread_ctx)
{
    int outstanding = 0;
    long i;

    for (i = 0; i < test_ctx.iters; ++i) {
        ++outstanding;
        if (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_NONE) {
            shmem_getmem_nbi(thread_ctx->local, thread_ctx->remote,
                             test_ctx.message_size, test_ctx.dst_pe);
            if (outstanding > test_ctx.nb_window) {
                shmem_quiet();
                outstanding = 0;
            }
        } else {
            shmem_ctx_getmem_nbi(thread_ctx->shmem_ctx,
                                 thread_ctx->local, thread_ctx->remote,
                                 test_ctx.message_size, test_ctx.dst_pe);
            if (outstanding > test_ctx.nb_window) {
                shmem_ctx_quiet(thread_ctx->shmem_ctx);
                outstanding = 0;
            }
        }
    }
}

static test_func_t test_funcs[] = {
    [TEST_PUT]     = do_put,
    [TEST_GET]     = do_get,
    [TEST_GET_NBI] = do_get_nbi,
};

static int set_affinity(thread_ctx_t *thread_ctx)
{
    cpu_set_t cpuset;
    int ret;

    __CPU_ZERO_S(sizeof(cpuset), &cpuset);
    __CPU_SET_S(thread_ctx->thread_num * 2 + 0, sizeof(cpuset), &cpuset); // TODO

    ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (ret) {
        printf("pthread_setaffinity_np() returned %d: %m\n", ret);
    }
    return ret;
}

static int create_shmem_ctx(thread_ctx_t *thread_ctx)
{
    long options;
    int ret;

    if (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_NONE) {
        thread_ctx->shmem_ctx = NULL;
        return 0;
    }

    options = 0;
    if (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_PRIVATE) {
        options |= SHMEM_CTX_PRIVATE;
    }

    ret = shmem_ctx_create(options, &thread_ctx->shmem_ctx);
    if (ret) {
        printf("shmem_ctx_create() failed\n");
        return -1;
    }

    return 0;
}

static void *thread_func(void *arg)
{
    thread_ctx_t *thread_ctx = arg;
    struct timeval tv_start, tv_end, tv_elapsed;
    double mr;
    int ret;

    ret = set_affinity(thread_ctx);
    if (ret) {
        return NULL;
    }

    ret = create_shmem_ctx(thread_ctx);
    if (ret) {
        return NULL;
    }

    thread_ctx->remote = test_ctx.buffer +
                         (thread_ctx->thread_num * test_ctx.offset);
    thread_ctx->local  = thread_ctx->remote;

    for (;;) {
        gettimeofday(&tv_start, NULL);
        pthread_barrier_wait(&test_ctx.barrier);

        /* run selected test */
        test_funcs[test_ctx.operation](thread_ctx);

        /* quiet */
        if (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_NONE) {
            shmem_quiet();
        } else {
            shmem_ctx_quiet(thread_ctx->shmem_ctx);
        }

        ret = pthread_barrier_wait(&test_ctx.barrier);
        if (ret == -1) {
            gettimeofday(&tv_end, NULL);
            timersub(&tv_end, &tv_start, &tv_elapsed);

            mr = /* total messages */  (test_ctx.num_threads * test_ctx.iters) /
                 /* total time */      (tv_elapsed.tv_sec + tv_elapsed.tv_usec * 1e-6) /
                 /* convert to Mpps */ 1e6;

            _printf("Total MR: %.2f Mpps  (%.2f per thread)", mr, mr / test_ctx.num_threads);
        }
    }

    if (thread_ctx->shmem_ctx != NULL) {
        shmem_ctx_destroy(thread_ctx->shmem_ctx);
    }
}

static void usage()
{
    printf("\n");
    printf("Usage: shmem_mt [options] <test_name>\n");
    printf("\n");
    printf("  test_name is one of: put, get, get_nbi\n");
    printf("\n");
    printf("  Options are:\n");
    printf("    -t <num>      Number of threads per PE (%d)\n", test_ctx.num_threads);
    printf("    -i <num>      Number of test iterations (%ld)\n", test_ctx.iters);
    printf("    -w <num>      Non-blocking window size (# of outstanding ops) (%d)\n",
           test_ctx.nb_window);
    printf("    -c <type>     SHMEM context type\n");
    printf("                     none    - no usage of contexts\n");
    printf("                     private - each thread creates a private context\n");
    printf("                     shared  - each thread creates a shared context\n");
    printf("    -s <size>     Message size (%zu)\n", test_ctx.message_size);
    printf("    -o <offset>   Distance between buffers of different threads (%zu)\n",
           test_ctx.offset);
    printf("    -h            Print this help message\n");
    printf("\n");
}

static int parse_args(int argc, char **argv)
{
    int c;

    while ( (c = getopt(argc, argv, "t:i:w:c:s:o:h")) != -1 ) {
        switch (c) {
        case 't':
            test_ctx.num_threads = atoi(optarg);
            break;
        case 'i':
            test_ctx.iters = strtol(optarg, NULL, 10);
            break;
        case 'w':
            test_ctx.nb_window = atoi(optarg);
            break;
        case 'c':
            if (!strcasecmp(optarg, "none")) {
                test_ctx.shmem_ctx_type = SHMEM_CTX_TYPE_NONE;
            } else if (!strcasecmp(optarg, "private")) {
                test_ctx.shmem_ctx_type = SHMEM_CTX_TYPE_PRIVATE;
            } else if (!strcasecmp(optarg, "shared")) {
                test_ctx.shmem_ctx_type = SHMEM_CTX_TYPE_SHARED;
            } else {
                printf("invalid argument for shmem context type: '%s'\n", optarg);
                usage();
                return -1;
            }
            break;
        case 's':
            test_ctx.message_size = strtol(optarg, NULL, 0);
            break;
        case 'o':
            test_ctx.offset = strtol(optarg, NULL, 0);
            break;
        case 'h':
            usage();
            return 0;
        default:
            usage();
            return -1;
        }
    }

    if (optind < argc) {
        if (!strcasecmp(argv[optind], "put")) {
            test_ctx.operation = TEST_PUT;
        } else if (!strcasecmp(argv[optind], "get")) {
            test_ctx.operation = TEST_GET;
        } else if (!strcasecmp(argv[optind], "get_nbi")) {
            test_ctx.operation = TEST_GET_NBI;
        } else {
            printf("invalid argument for test operation: '%s'\n", argv[optind]);
            usage();
            return -1;
        }
    }

    if (test_ctx.operation == TEST_NONE) {
        printf("test name not specified\n");
        usage();
        return -1;
    }

    return 0;
}

static void print_config()
{
    _printf("Configuration:");

    switch (test_ctx.operation) {
    case TEST_PUT:
        _printf("             Mode: shmem_put() in a loop");
        break;
    case TEST_GET:
        _printf("             Mode: shmem_get() in a loop");
        break;
    case TEST_GET_NBI:
        _printf("             Mode : loop { %d x shmem_get_nbi() + shmem_quiet() } ",
                test_ctx.nb_window);
        break;
    default:
        break;
    }

    _printf("        Total PEs : %d", shmem_n_pes());
    _printf("   Threads per PE : %d", test_ctx.num_threads);
    _printf("     Message size : %zu", test_ctx.message_size);
    _printf("       Iterations : %ld", test_ctx.iters);
    _printf("   SHMEM Contexts : %s",
            (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_NONE)    ? "not using contexts" :
            (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_PRIVATE) ? "private" :
            (test_ctx.shmem_ctx_type == SHMEM_CTX_TYPE_SHARED)  ? "shared" :
            "<invalid>");
    _printf("  Buffer distance : %zu", test_ctx.offset);
    _printf("");
}

int main(int argc, char **argv)
{
    thread_ctx_t *threads;
    int mt_provided;
    int ret;
    int i;

    ret = parse_args(argc, argv);
    if (ret < 0) {
        return ret;
    }

    shmem_init_thread((test_ctx.num_threads > 1) ? SHMEM_THREAD_MULTIPLE :
                                                   SHMEM_THREAD_SINGLE,
                      &mt_provided);

    if (shmem_n_pes() != 2) {
        _printf("the test should be run with 2 PEs");
        return -1;
    }

    print_config();

    test_ctx.dst_pe = 1;
    test_ctx.buffer = shmem_align(4096,
                                  (test_ctx.num_threads * test_ctx.offset) +
                                  test_ctx.message_size);

    if (shmem_my_pe() == 0) {
        pthread_barrier_init(&test_ctx.barrier, NULL, test_ctx.num_threads);
        threads = alloca(sizeof(*threads) * test_ctx.num_threads);
        for (i = 0; i < test_ctx.num_threads; ++i) {
            threads[i].thread_num = i;
        }

        if (test_ctx.num_threads == 1) {
            threads[0].pthread = pthread_self();
            thread_func(&threads[0]);
        } else {
            for (i = 0; i < test_ctx.num_threads; ++i) {
                pthread_create(&threads[i].pthread, NULL, thread_func, &threads[i]);
            }

            for (i = 0; i < test_ctx.num_threads; ++i) {
                void *result;
                pthread_join(threads[i].pthread, &result);
            }
        }
    }

    shmem_barrier_all();
    return 0;
}
