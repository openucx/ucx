/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#define _GNU_SOURCE /* For CPUSET_SIZE */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf.h"

#include <ucs/sys/sys.h>
#include <ucs/debug/log.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <netdb.h>
#include <getopt.h>
#include <string.h>
#include <sys/types.h>
#include <locale.h>
#if HAVE_MPI
#  include <mpi.h>
#elif HAVE_RTE
#   include<rte.h>
#endif

#define MAX_BATCH_FILES  32


enum {
    TEST_FLAG_PRINT_RESULTS = UCS_BIT(0),
    TEST_FLAG_PRINT_TEST    = UCS_BIT(1),
    TEST_FLAG_SET_AFFINITY  = UCS_BIT(8),
    TEST_FLAG_NUMERIC_FMT   = UCS_BIT(9),
    TEST_FLAG_PRINT_FINAL   = UCS_BIT(10),
    TEST_FLAG_PRINT_CSV     = UCS_BIT(11)
};

typedef struct sock_rte_group {
    int                          is_server;
    int                          connfd;
    void                         *self;
    size_t                       self_size;
} sock_rte_group_t;


typedef struct test_type {
    const char                   *name;
    ucx_perf_api_t               api;
    ucx_perf_cmd_t               command;
    ucx_perf_test_type_t         test_type;
    const char                   *desc;
} test_type_t;


struct perftest_context {
    ucx_perf_params_t            params;
    const char                   *server_addr;
    int                          port;
    unsigned                     cpu;
    unsigned                     flags;

    unsigned                     num_batch_files;
    char                         *batch_files[MAX_BATCH_FILES];
    char                         *test_names[MAX_BATCH_FILES];

    sock_rte_group_t             sock_rte_group;
};

#define TEST_PARAMS_ARGS   "t:n:s:W:O:w:D:H:oqT:"


test_type_t tests[] = {
    {"am_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
     "active message latency"},

    {"put_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
     "put latency"},

    {"add_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_PINGPONG,
     "atomic add latency"},

    {"get", UCX_PERF_API_UCT, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "get latency / bandwidth / message rate"},

    {"fadd", UCX_PERF_API_UCT, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic fetch-and-add latency / message rate"},

    {"swap", UCX_PERF_API_UCT, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic swap latency / message rate"},

    {"cswap", UCX_PERF_API_UCT, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic compare-and-swap latency / message rate"},

    {"am_bw", UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "active message bandwidth / message rate"},

    {"put_bw", UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "put bandwidth / message rate"},

    {"add_mr", UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic add message rate"},

    {"tag_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
     "tag match latency"},

    {"tag_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "tag match bandwidth"},

    {NULL}
};

static int safe_send(int sock, void *data, size_t size)
{
    size_t total = 0;
    int ret;

    while (total < size) {
        ret = send(sock, (char*)data + total, size - total, 0);
        if (ret < 0) {
            ucs_error("send() failed: %m");
            return -1;
        }
        total += ret;
    }
    return 0;
}

static int safe_recv(int sock, void *data, size_t size)
{
    size_t total = 0;
    int ret;

    while (total < size) {
        ret = recv(sock, (char*)data + total, size - total, 0);
        if (ret < 0) {
            ucs_error("recv() failed: %m");
            return -1;
        }
        total += ret;
    }
    return 0;
}

static void print_progress(char **test_names, unsigned num_names,
                           ucx_perf_result_t *result, unsigned flags, int final)
{
    static const char *fmt_csv     =  "%.0f,%.3f,%.3f,%.3f,%.2f,%.2f,%.0f,%.0f\n";
    static const char *fmt_numeric =  "%'14.0f %9.3f %9.3f %9.3f %10.2f %10.2f %'11.0f %'11.0f\n";
    static const char *fmt_plain   =  "%14.0f %9.3f %9.3f %9.3f %10.2f %10.2f %11.0f %11.0f\n";
    unsigned i;

    if (!(flags & TEST_FLAG_PRINT_RESULTS) ||
        (!final && (flags & TEST_FLAG_PRINT_FINAL)))
    {
        return;
    }

    if (flags & TEST_FLAG_PRINT_CSV) {
        for (i = 0; i < num_names; ++i) {
            printf("%s,", test_names[i]);
        }
    }

    printf((flags & TEST_FLAG_PRINT_CSV)   ? fmt_csv :
           (flags & TEST_FLAG_NUMERIC_FMT) ? fmt_numeric :
                                             fmt_plain,
           (double)result->iters,
           result->latency.typical * 1000000.0,
           result->latency.moment_average * 1000000.0,
           result->latency.total_average * 1000000.0,
           result->bandwidth.moment_average / (1024.0 * 1024.0),
           result->bandwidth.total_average / (1024.0 * 1024.0),
           result->msgrate.moment_average,
           result->msgrate.total_average);
    fflush(stdout);
}

static void print_header(struct perftest_context *ctx)
{
    const char *test_api_str;
    const char *test_data_str;
    test_type_t *test;
    unsigned i;

    if (ctx->flags & TEST_FLAG_PRINT_TEST) {
        for (test = tests; test->name; ++test) {
            if ((test->command == ctx->params.command) && (test->test_type == ctx->params.test_type)) {
                break;
            }
        }
        if (test->name != NULL) {
            if (test->api == UCX_PERF_API_UCT) {
                test_api_str = "transport layer";
                switch (ctx->params.uct.data_layout) {
                case UCT_PERF_DATA_LAYOUT_SHORT:
                    test_data_str = "short";
                    break;
                case UCT_PERF_DATA_LAYOUT_BCOPY:
                    test_data_str = "bcopy";
                    break;
                case UCT_PERF_DATA_LAYOUT_ZCOPY:
                    test_data_str = "zcopy";
                    break;
                default:
                    test_data_str = "(undefined)";
                    break;
                }
            } else if (test->api == UCX_PERF_API_UCP) {
                test_api_str = "protocol layer";
                test_data_str = "(automatic)"; /* TODO contig/stride/stream */
            } else {
                return;
            }

            printf("+------------------------------------------------------------------------------------------+\n");
            printf("| API:          %-60s               |\n", test_api_str);
            printf("| Test:         %-60s               |\n", test->desc);
            printf("| Data layout:  %-60s               |\n", test_data_str);
            printf("| Message size: %-60zu               |\n", ctx->params.message_size);
        }
    }

    if (ctx->flags & TEST_FLAG_PRINT_CSV) {
        if (ctx->flags & TEST_FLAG_PRINT_RESULTS) {
            for (i = 0; i < ctx->num_batch_files; ++i) {
                printf("%s,", basename(ctx->batch_files[i]));
            }
            printf("iterations,typical_lat,avg_lat,overall_lat,avg_bw,overall_bw,avg_mr,overall_mr\n");
        }
    } else {
        if (ctx->flags & TEST_FLAG_PRINT_RESULTS) {
            printf("+--------------+-----------------------------+---------------------+-----------------------+\n");
            printf("|              |       latency (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |\n");
            printf("+--------------+---------+---------+---------+----------+----------+-----------+-----------+\n");
            printf("| # iterations | typical | average | overall |  average |  overall |   average |   overall |\n");
            printf("+--------------+---------+---------+---------+----------+----------+-----------+-----------+\n");
        } else if (ctx->flags & TEST_FLAG_PRINT_TEST) {
            printf("+------------------------------------------------------------------------------------------+\n");
        }
    }
}

static void print_test_name(struct perftest_context *ctx)
{
    char buf[200];
    unsigned i, pos;

    if (!(ctx->flags & TEST_FLAG_PRINT_CSV) && (ctx->num_batch_files > 0)) {
        strcpy(buf, "+--------------+---------+---------+---------+----------+----------+-----------+-----------+");

        pos = 1;
        for (i = 0; i < ctx->num_batch_files; ++i) {
           if (i != 0) {
               buf[pos++] = '/';
           }
           memcpy(&buf[pos], ctx->test_names[i],
                  ucs_min(strlen(ctx->test_names[i]), sizeof(buf) - pos - 1));
           pos += strlen(ctx->test_names[i]);
        }

        if (ctx->flags & TEST_FLAG_PRINT_RESULTS) {
            printf("%s\n", buf);
        }
    }
}

static void usage(struct perftest_context *ctx, const char *program)
{
    test_type_t *test;

    printf("Usage: %s [ server-hostname ] [ options ]\n", program);
    printf("\n");
#if HAVE_MPI
    printf("This test can be also launched as an MPI application\n");
#elif HAVE_RTE
    printf("This test can be also launched as an libRTE application\n");
#endif
    printf("  Common options:\n");
    printf("\n");
    printf("  Test options:\n");
    printf("     -t <test>      Test to run.\n");
    for (test = tests; test->name; ++test) {
        printf("                   %10s : %s.\n", test->name, test->desc);
    }
    printf("\n");
    printf("     -D <layout>    Data layout.\n");
    printf("                        short : Use short messages API (cannot used for get).\n");
    printf("                        bcopy : Use copy-out API (cannot used for atomics).\n");
    printf("                        zcopy : Use zero-copy API (cannot used for atomics).\n");
    printf("\n");
    printf("     -d <device>    Device to use for testing.\n");
    printf("     -x <tl>        Transport to use for testing.\n");
    printf("     -c <cpu>       Set affinity to this CPU. (off)\n");
    printf("     -n <iters>     Number of iterations to run. (%ld)\n", ctx->params.max_iter);
    printf("     -s <size>      Message size. (%zu)\n", ctx->params.message_size);
    printf("     -H <size>      AM Header size. (%zu)\n", ctx->params.am_hdr_size);
    printf("     -w <iters>     Number of warm-up iterations. (%zu)\n", ctx->params.warmup_iter);
    printf("     -W <count>     Flow control window size, for active messages. (%u)\n", ctx->params.uct.fc_window);
    printf("     -O <count>     Maximal number of uncompleted outstanding sends. (%u)\n", ctx->params.max_outstanding);
    printf("     -N             Use numeric formatting - thousands separator.\n");
    printf("     -f             Print only final numbers.\n");
    printf("     -v             Print CSV-formatted output.\n");
    printf("     -p <port>      TCP port to use for data exchange. (%d)\n", ctx->port);
    printf("     -b <batchfile> Batch mode. Read and execute tests from a file.\n");
    printf("                       Every line of the file is a test to run. The first word is the\n");
    printf("                       test name, and the rest are command-line arguments for the test.\n");
    printf("     -T <thread>    Thread support level for progress engine (single).\n");
    printf("                     (The test itself is always single-threaded).\n");
    printf("                        single   : Single thread can access.\n");
    printf("                        funneled : One thread can access at a time.\n");
    printf("                        multi    : Multiple threads can access.\n");
    printf("     -h             Show this help message.\n");
    printf("\n");
    printf("  Server options:\n");
    printf("     -l             Accept clients in an infinite loop\n");
    printf("\n");
}

const char *__basename(const char *path)
{
    const char *p = strrchr(path, '/');
    return (p == NULL) ? path : p;
}

static void init_test_params(ucx_perf_params_t *params)
{
    params->api             = UCX_PERF_API_LAST;
    params->command         = UCX_PERF_CMD_LAST;
    params->test_type       = UCX_PERF_TEST_TYPE_LAST;
    params->thread_mode     = UCS_THREAD_MODE_SINGLE;
    params->wait_mode       = UCX_PERF_WAIT_MODE_LAST;
    params->max_outstanding = 1;
    params->warmup_iter     = 10000;
    params->message_size    = 8;
    params->am_hdr_size     = 8;
    params->alignment       = ucs_get_page_size();
    params->max_iter        = 1000000l;
    params->max_time        = 0.0;
    params->report_interval = 1.0;
    params->flags           = UCX_PERF_TEST_FLAG_VERBOSE;
    params->uct.fc_window   = UCT_PERF_TEST_MAX_FC_WINDOW;
    params->uct.data_layout = UCT_PERF_DATA_LAYOUT_SHORT;
}

static ucs_status_t parse_test_params(ucx_perf_params_t *params, char opt, const char *optarg)
{
    test_type_t *test;

    switch (opt) {
    case 't':
        for (test = tests; test->name; ++test) {
            if (!strcmp(optarg, test->name)) {
                params->api       = test->api;
                params->command   = test->command;
                params->test_type = test->test_type;
                break;
            }
        }
        if (test->name == NULL) {
            ucs_error("Invalid option argument for -t");
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;
    case 'D':
        if (0 == strcmp(optarg, "short")) {
            params->uct.data_layout   = UCT_PERF_DATA_LAYOUT_SHORT;
        } else if (0 == strcmp(optarg, "bcopy")) {
            params->uct.data_layout   = UCT_PERF_DATA_LAYOUT_BCOPY;
        } else if (0 == strcmp(optarg, "zcopy")) {
            params->uct.data_layout   = UCT_PERF_DATA_LAYOUT_ZCOPY;
        } else {
            ucs_error("Invalid option argument for -D");
            return -1;
        }
        return UCS_OK;
    case 'n':
        params->max_iter = atol(optarg);
        return UCS_OK;
    case 's':
        params->message_size = atol(optarg);
        return UCS_OK;
    case 'H':
        params->am_hdr_size = atol(optarg);
        return UCS_OK;
    case 'W':
        params->uct.fc_window = atoi(optarg);
        return UCS_OK;
    case 'O':
        params->max_outstanding = atoi(optarg);
        return UCS_OK;
    case 'w':
        params->warmup_iter = atol(optarg);
        return UCS_OK;
    case 'o':
        params->flags |= UCX_PERF_TEST_FLAG_ONE_SIDED;
        return UCS_OK;
    case 'q':
        params->flags &= ~UCX_PERF_TEST_FLAG_VERBOSE;
        return UCS_OK;
    case 'T':
        if (0 == strcmp(optarg, "single")) {
            params->thread_mode = UCS_THREAD_MODE_SINGLE;
            return UCS_OK;
        } else if (0 == strcmp(optarg, "funneled")) {
            params->thread_mode = UCS_THREAD_MODE_FUNNELED;
            return UCS_OK;
        } else if (0 == strcmp(optarg, "multi")) {
            params->thread_mode = UCS_THREAD_MODE_MULTI;
            return UCS_OK;
        } else {
            ucs_error("Invalid option argument for -T");
            return UCS_ERR_INVALID_PARAM;
        }
    default:
       return UCS_ERR_INVALID_PARAM;
    }
}

static ucs_status_t read_batch_file(FILE *batch_file, ucx_perf_params_t *params,
                                    char** test_name_p)
{
#define MAX_SIZE 256
    ucs_status_t status;
    char buf[MAX_SIZE];
    int argc;
    char *argv[MAX_SIZE + 1];
    char c, *p;

    do {
        if (fgets(buf, sizeof(buf) - 1, batch_file) == NULL) {
            return UCS_ERR_NO_ELEM;
        }

        argc = 0;
        p = strtok(buf, " \t\n\r");
        while (p && (argc < MAX_SIZE)) {
            argv[argc++] = p;
            p = strtok(NULL, " \t\n\r");
        }
        argv[argc] = NULL;
    } while ((argc == 0) || (argv[0][0] == '#'));


    optind = 1;
    while ((c = getopt (argc, argv, TEST_PARAMS_ARGS)) != -1) {
        status = parse_test_params(params, c, optarg);
        if (status != UCS_OK) {
            ucs_error("Invalid argument in batch file: -%c", c);
            return status;
        }
    }

    *test_name_p = strdup(argv[0]);
    return UCS_OK;
}

static ucs_status_t parse_opts(struct perftest_context *ctx, int argc, char **argv)
{
    ucs_status_t status;
    char c;

    init_test_params(&ctx->params);
    ctx->server_addr            = NULL;
    ctx->num_batch_files        = 0;
    ctx->port                   = 13337;
    ctx->flags                  = 0;
    strcpy(ctx->params.uct.dev_name, "");
    strcpy(ctx->params.uct.tl_name, "");

    optind = 1;
    while ((c = getopt (argc, argv, "p:d:x:b:Nfvc:h" TEST_PARAMS_ARGS)) != -1) {
        switch (c) {
        case 'p':
            ctx->port = atoi(optarg);
            break;
        case 'd':
            ucs_snprintf_zero(ctx->params.uct.dev_name, sizeof(ctx->params.uct.dev_name),
                              "%s", optarg);
            break;
        case 'x':
            ucs_snprintf_zero(ctx->params.uct.tl_name, sizeof(ctx->params.uct.tl_name),
                              "%s", optarg);
            break;
        case 'b':
            if (ctx->num_batch_files < MAX_BATCH_FILES) {
                ctx->batch_files[ctx->num_batch_files++] = strdup(optarg);
            }
            break;
        case 'N':
            ctx->flags |= TEST_FLAG_NUMERIC_FMT;
            break;
        case 'f':
            ctx->flags |= TEST_FLAG_PRINT_FINAL;
            break;
        case 'v':
            ctx->flags |= TEST_FLAG_PRINT_CSV;
            break;
        case 'c':
            ctx->flags |= TEST_FLAG_SET_AFFINITY;
            ctx->cpu = atoi(optarg);
            break;
        case 'h':
            usage(ctx, __basename(argv[0]));
            return UCS_ERR_CANCELED;
        default:
            status = parse_test_params(&ctx->params, c, optarg);
            if (status != UCS_OK) {
                usage(ctx, __basename(argv[0]));
                return status;
            }
            break;
        }
    }

    if (optind < argc) {
        ctx->server_addr   = argv[optind];
    }

    return UCS_OK;
}

static ucs_status_t validate_params(struct perftest_context *ctx)
{
    ucs_status_t status;
    uct_resource_desc_t *resources;
    unsigned num_resources;

    if (ctx->params.api == UCX_PERF_API_UCT) {
        status = uct_query_resources(ctx->params.uct.context, &resources,
                                     &num_resources);
        if (status != UCS_OK) {
            ucs_error("Failed to query resources");
            goto error;
        }

        if (!strlen(ctx->params.uct.tl_name)) {
            if (num_resources <= 0) {
                ucs_error("Must specify transport");
                status = UCS_ERR_INVALID_PARAM;
                goto error;
            }
            strncpy(ctx->params.uct.tl_name, resources[0].tl_name,
                    sizeof(ctx->params.uct.tl_name));
            ucs_info("No transport was specified, selecting %s\n", ctx->params.uct.tl_name);
        }

        if (!strlen(ctx->params.uct.dev_name)) {
            int i;
            if (num_resources <= 0) {
                ucs_error("No specify device name");
                status = UCS_ERR_INVALID_PARAM;
                goto error;
            }
            for (i = 0; i < num_resources; i++) {
                if(!strcmp(ctx->params.uct.tl_name, resources[i].tl_name)) {
                    strncpy(ctx->params.uct.dev_name, resources[i].dev_name,
                            sizeof(ctx->params.uct.dev_name));
                    ucs_info("No device was specified, selecting %s\n", ctx->params.uct.dev_name);
                }
            }
            if (!strlen(ctx->params.uct.dev_name)) {
                ucs_error("Device for transport %s was not found", ctx->params.uct.tl_name);
                status = UCS_ERR_INVALID_PARAM;
                goto error;
            }
        }
    error:
        uct_release_resource_list(resources);
    } else {
        status = UCS_OK;
    }
    return status;
}

unsigned sock_rte_group_size(void *rte_group)
{
    return 2;
}

unsigned sock_rte_group_index(void *rte_group)
{
    sock_rte_group_t *group = rte_group;
    return group->is_server ? 0 : 1;
}

void sock_rte_barrier(void *rte_group)
{
    sock_rte_group_t *group = rte_group;
    const unsigned magic = 0xdeadbeef;
    unsigned sync;

    sync = magic;
    safe_send(group->connfd, &sync, sizeof(unsigned));

    sync = 0;
    safe_recv(group->connfd, &sync, sizeof(unsigned));

    ucs_assert(sync == magic);
}

void sock_rte_send(void *rte_group, unsigned dest, void *value, size_t size)
{
    sock_rte_group_t *group = rte_group;
    unsigned me = sock_rte_group_index(rte_group);

    if (dest == me) {
        group->self = realloc(group->self, group->self_size + size);
        memcpy(group->self + group->self_size, value, size);
        group->self_size += size;
    } else if (dest == 1 - me) {
        safe_send(group->connfd, value, size);
    }
}

void sock_rte_recv(void *rte_group, unsigned src, void *value, size_t size)
{
    sock_rte_group_t *group = rte_group;
    unsigned me = sock_rte_group_index(rte_group);
    void *prev_self;

    if (src == me) {
        ucs_assert(group->self_size >= size);
        memcpy(value, group->self, size);
        group->self_size -= size;

        prev_self = group->self;
        group->self = malloc(group->self_size);
        memcpy(group->self, prev_self + size, group->self_size);
    } else if (src == 1 - me) {
        safe_recv(group->connfd, value, size);
    }
}

void sock_rte_post_vec(void *rte_group, struct iovec * iovec, size_t num, void **req)
{
    int i, j;
    int group_size;
    int group_index;

    group_size = sock_rte_group_size(rte_group);
    group_index = sock_rte_group_index(rte_group);

    for (i = 0; i < group_size; ++i) {
        if (i != group_index) {
            for (j = 0; j < num; ++j) {
                sock_rte_send(rte_group, i, iovec[j].iov_base, iovec[j].iov_len);
            }
        }
    }
}

void sock_rte_recv_vec(void *rte_group, unsigned dest, struct iovec *iovec, size_t num, void * req)
{
    int i, group_index;

    group_index = sock_rte_group_index(rte_group);
    if (dest != group_index) {
        for (i = 0; i < num; ++i) {
            sock_rte_recv(rte_group, dest, iovec[i].iov_base, iovec[i].iov_len);
        }
    }
}

void sock_rte_exchange_vec(void *rte_group, void * req)
{
}

static void sock_rte_report(void *rte_group, ucx_perf_result_t *result, int is_final)
{
    struct perftest_context *ctx =
                    ucs_container_of(rte_group, struct perftest_context, sock_rte_group);
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags, is_final);
}

ucx_perf_rte_t sock_rte = {
    .group_size    = sock_rte_group_size,
    .group_index   = sock_rte_group_index,
    .barrier       = sock_rte_barrier,
    .post_vec      = sock_rte_post_vec,
    .recv_vec      = sock_rte_recv_vec,
    .exchange_vec  = sock_rte_exchange_vec,
    .report        = sock_rte_report,
};

ucs_status_t setup_sock_rte(struct perftest_context *ctx)
{
    struct sockaddr_in inaddr;
    struct hostent *he;
    ucs_status_t status;
    int optval = 1;
    int sockfd, connfd;
    int ret;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        ucs_error("socket() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    if (ctx->server_addr == NULL) {
        optval = 1;
        ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
        if (ret < 0) {
            ucs_error("setsockopt(SO_REUSEADDR) failed: %m");
            status = UCS_ERR_INVALID_PARAM;
            goto err_close_sockfd;
        }

        inaddr.sin_family      = AF_INET;
        inaddr.sin_port        = htons(ctx->port);
        inaddr.sin_addr.s_addr = INADDR_ANY;
        memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
        ret = bind(sockfd, (struct sockaddr*)&inaddr, sizeof(inaddr));
        if (ret < 0) {
            ucs_error("bind() failed: %m");
            status = UCS_ERR_INVALID_ADDR;
            goto err_close_sockfd;
        }

        ret = listen(sockfd, 10);
        if (ret < 0) {
            ucs_error("listen() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_close_sockfd;
        }

        printf("Waiting for connection...\n");

        /* Accept next connection */
        connfd = accept(sockfd, NULL, NULL);
        if (connfd < 0) {
            ucs_error("accept() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_close_sockfd;
        }

        close(sockfd);
        safe_recv(connfd, &ctx->params, sizeof(ctx->params));

        ctx->sock_rte_group.connfd    = connfd;
        ctx->sock_rte_group.is_server = 1;

    } else {
        status = validate_params(ctx);
        if (status != UCS_OK) {
            goto err_close_sockfd;
        }

        he = gethostbyname(ctx->server_addr);
        if (he == NULL || he->h_addr_list == NULL) {
            ucs_error("host %s not found: %s", ctx->server_addr,
                      hstrerror(h_errno));
            status = UCS_ERR_INVALID_ADDR;
            goto err_close_sockfd;
        }

        inaddr.sin_family = he->h_addrtype;
        inaddr.sin_port   = htons(ctx->port);
        ucs_assert(he->h_length == sizeof(inaddr.sin_addr));
        memcpy(&inaddr.sin_addr, he->h_addr_list[0], he->h_length);
        memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));

        ret = connect(sockfd, (struct sockaddr*)&inaddr, sizeof(inaddr));
        if (ret < 0) {
            ucs_error("connect() failed: %m");
            status = UCS_ERR_UNREACHABLE;
            goto err_close_sockfd;
        }

        safe_send(sockfd, &ctx->params, sizeof(ctx->params));

        ctx->sock_rte_group.connfd    = sockfd;
        ctx->sock_rte_group.is_server = 0;
    }

    if (ctx->sock_rte_group.is_server) {
        ctx->flags |= TEST_FLAG_PRINT_TEST;
    } else {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }
    ctx->sock_rte_group.self      = NULL;
    ctx->sock_rte_group.self_size = 0;
    ctx->params.rte_group         = &ctx->sock_rte_group;
    ctx->params.rte               = &sock_rte;
    return UCS_OK;

err_close_sockfd:
    close(sockfd);
err:
    return status;
}

static ucs_status_t cleanup_sock_rte(struct perftest_context *ctx)
{
    close(ctx->sock_rte_group.connfd);
    free(ctx->sock_rte_group.self);
    return UCS_OK;
}

#if HAVE_MPI
unsigned mpi_rte_group_size(void *rte_group)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

unsigned mpi_rte_group_index(void *rte_group)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

void mpi_rte_barrier(void *rte_group)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

void mpi_rte_post_vec(void *rte_group, struct iovec * iovec, size_t num, void **req)
{
    int i, j;
    int group_size;
    int group_index;

    MPI_Comm_rank(MPI_COMM_WORLD, &group_index);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    for (i = 0; i < group_size; ++i) {
        if (i != group_index) {
            for (j = 0; j < num; ++j) {
                MPI_Send(iovec[j].iov_base, iovec[j].iov_len, MPI_CHAR, i,
                         1, MPI_COMM_WORLD);
            }
        }
    }
}

void mpi_rte_recv_vec(void *rte_group, unsigned dest, struct iovec *iovec, size_t num, void * req)
{
    int i, group_index;

    MPI_Comm_rank(MPI_COMM_WORLD, &group_index);
    if (dest != group_index) {
        for (i = 0; i < num; ++i) {
            MPI_Recv(iovec[i].iov_base, iovec[i].iov_len, MPI_CHAR, dest, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void mpi_rte_exchange_vec(void *rte_group, void * req)
{
    MPI_Barrier(MPI_COMM_WORLD);
}

static void mpi_rte_report(void *rte_group, ucx_perf_result_t *result, int is_final)
{
    struct perftest_context *ctx = rte_group;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags, is_final);
}

ucx_perf_rte_t mpi_rte = {
    .group_size    = mpi_rte_group_size,
    .group_index   = mpi_rte_group_index,
    .barrier       = mpi_rte_barrier,
    .post_vec      = mpi_rte_post_vec,
    .recv_vec      = mpi_rte_recv_vec,
    .exchange_vec  = mpi_rte_exchange_vec,
    .report        = mpi_rte_report,
};
#elif HAVE_RTE
unsigned ext_rte_group_size(void *rte_group)
{
    return rte_group_size((rte_group_t)rte_group);
}

unsigned ext_rte_group_index(void *rte_group)
{
    return rte_group_rank((rte_group_t)rte_group);
}

void ext_rte_barrier(void *rte_group)
{
    int rc;
    rc = rte_barrier((rte_group_t)rte_group);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_barrier");
    }
}

void ext_rte_post_vec(void *rte_group, struct iovec* iovec, size_t num, void **req)
{
    int i, rc;
    rte_group_t group = (rte_group_t)rte_group;
    rte_srs_session_t session;
    rte_iovec_t *r_vec;

    rc = rte_srs_session_create(group, 0, &session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_session_create");
    }

    r_vec = calloc(num, sizeof(rte_iovec_t));
    if (r_vec == NULL) {
        return;
    }
    for (i = 0; i < num; ++i) {
        r_vec[i].iov_base = iovec[i].iov_base;
        r_vec[i].type = rte_datatype_uint8_t;
        r_vec[i].count = iovec[i].iov_len;
    }
    rc = rte_srs_set_data(session, "KEY_PERF", r_vec, num);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_set_data");
    }
    *req = session;
    free(r_vec);
}

void ext_rte_recv_vec(void *rte_group, unsigned dest, struct iovec *iovec, size_t num, void * req)
{
    rte_srs_session_t session = (rte_srs_session_t)req;
    rte_group_t group = (rte_group_t)rte_group;
    void *buffer = NULL;
    int size;
    uint32_t offset = 0;
    rte_iovec_t *r_vec;
    int i, rc;

    rc = rte_srs_get_data(session, rte_group_index_to_ec(group, dest),
                     "KEY_PERF", &buffer, &size);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_get_data");
    }
    r_vec = calloc(num, sizeof(rte_iovec_t));
    if (r_vec == NULL) {
        ucs_error("Failed to allocate memory");
        return;
    }
    for (i = 0; i < num; ++i) {
        r_vec[i].iov_base = iovec[i].iov_base;
        r_vec[i].type = rte_datatype_uint8_t;
        r_vec[i].count = iovec[i].iov_len;
        rte_unpack(&r_vec[i], buffer, &offset);
    }
    rc = rte_srs_session_destroy(session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_session_destroy");
    }
    free(buffer);
    free(r_vec);
}

void ext_rte_exchange_vec(void *rte_group, void * req)
{
    rte_srs_session_t session = (rte_srs_session_t)req;
    int rc;
    rc = rte_srs_exchange_data(session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_exchange_data");
    }
}

static void ext_rte_report(void *rte_group, ucx_perf_result_t *result, int is_final)
{
    struct perftest_context *ctx = rte_group;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags, is_final);
}

ucx_perf_rte_t ext_rte = {
    .group_size    = ext_rte_group_size,
    .group_index   = ext_rte_group_index,
    .barrier       = ext_rte_barrier,
    .report        = ext_rte_report,
    .post_vec      = ext_rte_post_vec,
    .recv_vec      = ext_rte_recv_vec,
    .exchange_vec  = ext_rte_exchange_vec,
};
#endif

static ucs_status_t setup_mpi_rte(struct perftest_context *ctx)
{
#if HAVE_MPI
    ucs_status_t status;
    int rank;

    status = validate_params(ctx);
    if (status != UCS_OK) {
        return status;;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->sock_rte_group.self      = NULL;
    ctx->sock_rte_group.self_size = 0;
    ctx->params.rte_group         = ctx;
    ctx->params.rte               = &mpi_rte;
#elif HAVE_RTE
    ucs_status_t status;
    rte_group_t group;

    status = validate_params(ctx);
    if (status != UCS_OK) {
        return status;;
    }

    rte_init(NULL, NULL, &group);
    if (0 == rte_group_rank(group)) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->sock_rte_group.self      = NULL;
    ctx->sock_rte_group.self_size = 0;
    ctx->params.rte_group         = (void *)group;
    ctx->params.rte               = &ext_rte;
#endif
    return UCS_OK;
}

static ucs_status_t cleanup_mpi_rte(struct perftest_context *ctx)
{
    return UCS_OK;
}

static ucs_status_t check_system(struct perftest_context *ctx)
{
    cpu_set_t cpuset;
    unsigned i, count, nr_cpus;
    int ret;

    ret = sysconf(_SC_NPROCESSORS_CONF);
    if (ret < 0) {
        ucs_error("failed to get local cpu count: %m");
        return UCS_ERR_INVALID_PARAM;
    }
    nr_cpus = ret;

    memset(&cpuset, 0, sizeof(cpuset));
    if (ctx->flags & TEST_FLAG_SET_AFFINITY) {
        if (ctx->cpu >= nr_cpus) {
            ucs_error("cpu (%u) ot of range (0..%u)", ctx->cpu, nr_cpus - 1);
            return UCS_ERR_INVALID_PARAM;
        }
        CPU_SET(ctx->cpu, &cpuset);

        ret = sched_setaffinity(0, sizeof(cpuset), &cpuset);
        if (ret) {
            ucs_warn("sched_setaffinity() failed: %m");
            return UCS_ERR_INVALID_PARAM;
        }
    } else {
        ret = sched_getaffinity(0, sizeof(cpuset), &cpuset);
        if (ret) {
            ucs_warn("sched_getaffinity() failed: %m");
            return UCS_ERR_INVALID_PARAM;
        }

        count = 0;
        for (i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                ++count;
            }
        }
        if (count > 2) {
            ucs_warn("CPU affinity is not set (bound to %u cpus)."
                            " Performance may be impacted.", count);
        }
    }

    return UCS_OK;
}

static ucs_status_t run_test_recurs(struct perftest_context *ctx,
                                    ucx_perf_params_t *parent_params,
                                    unsigned depth)
{
    ucx_perf_params_t params;
    ucx_perf_result_t result;
    ucs_status_t status;
    FILE *batch_file;

    if (depth >= ctx->num_batch_files) {
        print_test_name(ctx);
        return ucx_perf_run(parent_params, &result);
    }

    batch_file = fopen(ctx->batch_files[depth], "r");
    if (batch_file == NULL) {
        ucs_error("Failed to open '%s': %m", optarg);
        return UCS_ERR_IO_ERROR;
    }

    params = *parent_params;
    while ((status = read_batch_file(batch_file, &params, &ctx->test_names[depth])) == UCS_OK) {
        status = run_test_recurs(ctx, &params, depth + 1);
        free(ctx->test_names[depth]);
    }

    fclose(batch_file);
    return UCS_OK;
}

static ucs_status_t run_test(struct perftest_context *ctx)
{
    ucs_status_t status;

    setlocale(LC_ALL, "en_US");

    print_header(ctx);

    status = run_test_recurs(ctx, &ctx->params, 0);
    if (status != UCS_OK) {
        ucs_error("Failed to run test: %s", ucs_status_string(status));
    }

    return status;
}

static ucs_status_t open_context(struct perftest_context *ctx)
{
    ucp_config_t *config;
    ucs_status_t status;

    switch (ctx->params.api) {
    case UCX_PERF_API_UCT:
        return uct_init(&ctx->params.uct.context);
    case UCX_PERF_API_UCP:
        status = ucp_config_read(NULL, NULL, &config);
        if (status != UCS_OK) {
            return status;
        }

        status = ucp_init(config, 0, &ctx->params.ucp.context);
        ucp_config_release(config);
        return status;
    default:
        return UCS_ERR_INVALID_PARAM;
    }
}

static void close_context(struct perftest_context *ctx)
{
    switch (ctx->params.api) {
    case UCX_PERF_API_UCT:
        uct_cleanup(ctx->params.uct.context);
        return;
    case UCX_PERF_API_UCP:
        ucp_cleanup(ctx->params.ucp.context);
        return;
    default:
        return;
    }
}

int main(int argc, char **argv)
{
    struct perftest_context ctx;
    ucs_status_t status;
    int rte = 0;
    int ret;

#ifdef __COVERITY__
    rte = rand(); /* Shut up deadcode error */
#endif

#if HAVE_MPI
    /* Don't try MPI when running interactively */
    if (!isatty(0) && (MPI_Init(&argc, &argv) == 0)) {
        rte = 1;
    }
#elif HAVE_RTE
    rte = 1;
#endif

    /* Parse command line */
    if (parse_opts(&ctx, argc, argv) != UCS_OK) {
        ret = -127;
        goto out;
    }

    status = check_system(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out;
    }

    /* Create RTE */
    status = (rte) ? setup_mpi_rte(&ctx) : setup_sock_rte(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out;
    }

    status = open_context(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_close_context;
    }

    /* Run the test */
    status = run_test(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_cleanup_rte;
    }

    ret = 0;

out_close_context:
    close_context(&ctx);
out_cleanup_rte:
    (rte) ? cleanup_mpi_rte(&ctx) : cleanup_sock_rte(&ctx);
out:
    if (rte) {
#if HAVE_MPI
        MPI_Finalize();
#elif HAVE_RTE
        rte_finalize();
#endif
    }
    return ret;
}
