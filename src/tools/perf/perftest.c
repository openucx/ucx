/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University 
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf.h"
#include "libperf_int.h"

#include <ucs/sys/string.h>
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
#include <sys/poll.h>
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
    int                          mpi;
    unsigned                     cpu;
    unsigned                     flags;

    unsigned                     num_batch_files;
    char                         *batch_files[MAX_BATCH_FILES];
    char                         *test_names[MAX_BATCH_FILES];

    sock_rte_group_t             sock_rte_group;
};

#define TEST_PARAMS_ARGS   "t:n:s:W:O:w:D:i:H:oSCqMr:T:d:x:A:BUm:"


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
     "UCP tag match latency"},

    {"tag_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP tag match bandwidth"},

    {"tag_sync_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_PINGPONG,
     "UCP tag sync match latency"},

    {"tag_sync_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP tag sync match bandwidth"},

    {"ucp_put_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
     "UCP put latency"},

    {"ucp_put_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP put bandwidth"},

    {"ucp_get", UCX_PERF_API_UCP, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP get latency / bandwidth / message rate"},

    {"ucp_add", UCX_PERF_API_UCP, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP atomic add bandwidth / message rate"},

    {"ucp_fadd", UCX_PERF_API_UCP, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP atomic fetch-and-add latency / bandwidth / message rate"},

    {"ucp_swap", UCX_PERF_API_UCP, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP atomic swap latency / bandwidth / message rate"},

    {"ucp_cswap", UCX_PERF_API_UCP, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP atomic compare-and-swap latency / bandwidth / message rate"},

    {"stream_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "UCP stream bandwidth"},

    {"stream_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_PINGPONG,
     "UCP stream latency"},

     {NULL}
};

static int sock_io(int sock, ssize_t (*sock_call)(int, void *, size_t, int),
                   int poll_events, void *data, size_t size,
                   void (*progress)(void *arg), void *arg, const char *name)
{
    size_t total = 0;
    struct pollfd pfd;
    int ret;

    while (total < size) {
        pfd.fd      = sock;
        pfd.events  = poll_events;
        pfd.revents = 0;

        ret = poll(&pfd, 1, 1); /* poll for 1ms */
        if (ret > 0) {
            ucs_assert(ret == 1);
            ucs_assert(pfd.revents & poll_events);

            ret = sock_call(sock, (char*)data + total, size - total, 0);
            if (ret < 0) {
                ucs_error("%s() failed: %m", name);
                return -1;
            }
            total += ret;
        } else if ((ret < 0) && (errno != EINTR)) {
            ucs_error("poll(fd=%d) failed: %m", sock);
            return -1;
        }

        /* progress user context */
        if (progress != NULL) {
            progress(arg);
        }
    }
    return 0;
}

static int safe_send(int sock, void *data, size_t size,
                     void (*progress)(void *arg), void *arg)
{
    return sock_io(sock, (void*)send, POLLOUT, data, size, progress, arg, "send");
}

static int safe_recv(int sock, void *data, size_t size,
                     void (*progress)(void *arg), void *arg)
{
    return sock_io(sock, recv, POLLIN, data, size, progress, arg, "recv");
}

static void print_progress(char **test_names, unsigned num_names,
                           const ucx_perf_result_t *result, unsigned flags,
                           int final)
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
            printf("| Message size: %-60zu               |\n", ucx_perf_get_message_size(&ctx->params));
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

static void usage(const struct perftest_context *ctx, const char *program)
{
    test_type_t *test;
    int UCS_V_UNUSED rank;

#if HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ctx->mpi && (rank != 0)) {
        return;
    }
#endif

#if HAVE_MPI
    printf("  Note: test can be also launched as an MPI application\n");
    printf("\n");
#elif HAVE_RTE
    printf("  Note: this test can be also launched as an libRTE application\n");
    printf("\n");
#endif
    printf("  Usage: %s [ server-hostname ] [ options ]\n", program);
    printf("\n");
    printf("  Common options:\n");
    printf("     -t <test>      test to run:\n");
    for (test = tests; test->name; ++test) {
        printf("                   %11s - %s\n", test->name, test->desc);
    }
    printf("     -s <size>      list of scatter-gather sizes for single message (%zu)\n",
                                ctx->params.msg_size_list[0]);
    printf("                    for example: \"-s 16,48,8192,8192,14\"\n");
    printf("     -n <iters>     number of iterations to run (%ld)\n", ctx->params.max_iter);
    printf("     -w <iters>     number of warm-up iterations (%zu)\n",
                                ctx->params.warmup_iter);
    printf("     -c <cpu>       set affinity to this CPU (off)\n");
    printf("     -O <count>     maximal number of uncompleted outstanding sends (%u)\n",
                                ctx->params.max_outstanding);
    printf("     -i <offset>    distance between consecutive scatter-gather entries (%zu)\n",
                                ctx->params.iov_stride);
    printf("     -T <threads>   number of threads in the test (%d), if >1 implies \"-M multi\" for UCP\n",
                                ctx->params.thread_count);
    printf("     -B             register memory with NONBLOCK flag\n");
    printf("     -b <file>      read and execute tests from a batch file: every line in the\n");
    printf("                    file is a test to run, first word is test name, the rest of\n");
    printf("                    the line is command-line arguments for the test.\n");
    printf("     -p <port>      TCP port to use for data exchange (%d)\n", ctx->port);
#if HAVE_MPI
    printf("     -P <0|1>       disable/enable MPI mode (%d)\n", ctx->mpi);
#endif
    printf("     -m <mem type>  memory type of messages\n");
    printf("                        host - system memory(default)\n");
#if HAVE_CUDA
    printf("                        cuda - NVIDIA GPU memory\n");
#endif
    printf("     -h             show this help message\n");
    printf("\n");
    printf("  Output format:\n");
    printf("     -N             use numeric formatting (thousands separator)\n");
    printf("     -f             print only final numbers\n");
    printf("     -v             print CSV-formatted output\n");
    printf("\n");
    printf("  UCT only:\n");
    printf("     -d <device>    device to use for testing\n");
    printf("     -x <tl>        transport to use for testing\n");
    printf("     -D <layout>    data layout for sender side:\n");
    printf("                        short - short messages API (default, cannot be used for get)\n");
    printf("                        bcopy - copy-out API (cannot be used for atomics)\n");
    printf("                        zcopy - zero-copy API (cannot be used for atomics)\n");
    printf("                        iov    - scatter-gather list (iovec)\n");
    printf("     -W <count>     flow control window size, for active messages (%u)\n",
                                ctx->params.uct.fc_window);
    printf("     -H <size>      active message header size (%zu)\n",
                                ctx->params.am_hdr_size);
    printf("     -A <mode>      asynchronous progress mode (thread)\n");
    printf("                        thread - separate progress thread\n");
    printf("                        signal - signal-based timer\n");
    printf("\n");
    printf("  UCP only:\n");
    printf("     -M <thread>    thread support level for progress engine (single)\n");
    printf("                        single     - only the master thread can access\n");
    printf("                        serialized - one thread can access at a time\n");
    printf("                        multi      - multiple threads can access\n");
    printf("     -D <layout>[,<layout>]\n");
    printf("                    data layout for sender and receiver side (contig)\n");
    printf("                        contig - Continuous datatype\n");
    printf("                        iov    - Scatter-gather list\n");
    printf("     -C             use wild-card tag for tag tests\n");
    printf("     -U             force unexpected flow by using tag probe\n");
    printf("     -r <mode>      receive mode for stream tests (recv)\n");
    printf("                        recv       : Use ucp_stream_recv_nb\n");
    printf("                        recv_data  : Use ucp_stream_recv_data_nb\n");
    printf("\n");
}

static const char *__basename(const char *path)
{
    const char *p = strrchr(path, '/');
    return (p == NULL) ? path : (p + 1);
}

static ucs_status_t parse_ucp_datatype_params(const char *optarg,
                                              ucp_perf_datatype_t *datatype)
{
    const char  *iov_type         = "iov";
    const size_t iov_type_size    = strlen("iov");
    const char  *contig_type      = "contig";
    const size_t contig_type_size = strlen("contig");

    if (0 == strncmp(optarg, iov_type, iov_type_size)) {
        *datatype = UCP_PERF_DATATYPE_IOV;
    } else if (0 == strncmp(optarg, contig_type, contig_type_size)) {
        *datatype = UCP_PERF_DATATYPE_CONTIG;
    } else {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t parse_message_sizes_params(const char *optarg,
                                               ucx_perf_params_t *params)
{
    char *optarg_ptr, *optarg_ptr2;
    size_t token_num, token_it;
    const char delim = ',';

    optarg_ptr = (char *)optarg;
    token_num  = 0;
    /* count the number of given message sizes */
    while ((optarg_ptr = strchr(optarg_ptr, delim)) != NULL) {
        ++optarg_ptr;
        ++token_num;
    }
    ++token_num;

    free(params->msg_size_list); /* free previously allocated buffer */
    params->msg_size_list = malloc(sizeof(*params->msg_size_list) * token_num);
    if (NULL == params->msg_size_list) {
        return UCS_ERR_NO_MEMORY;
    }

    optarg_ptr = (char *)optarg;
    errno = 0;
    for (token_it = 0; token_it < token_num; ++token_it) {
        params->msg_size_list[token_it] = strtoul(optarg_ptr, &optarg_ptr2, 10);
        if (((ERANGE == errno) && (ULONG_MAX == params->msg_size_list[token_it])) ||
            ((errno != 0) && (params->msg_size_list[token_it] == 0)) ||
            (optarg_ptr == optarg_ptr2)) {
            free(params->msg_size_list);
            params->msg_size_list = NULL; /* prevent double free */
            ucs_error("Invalid option substring argument at position %lu", token_it);
            return UCS_ERR_INVALID_PARAM;
        }
        optarg_ptr = optarg_ptr2 + 1;
    }

    params->msg_size_cnt = token_num;
    return UCS_OK;
}

static void init_test_params(ucx_perf_params_t *params)
{
    params->api             = UCX_PERF_API_LAST;
    params->command         = UCX_PERF_CMD_LAST;
    params->test_type       = UCX_PERF_TEST_TYPE_LAST;
    params->thread_mode     = UCS_THREAD_MODE_SINGLE;
    params->thread_count    = 1;
    params->async_mode      = UCS_ASYNC_MODE_THREAD;
    params->wait_mode       = UCX_PERF_WAIT_MODE_LAST;
    params->max_outstanding = 1;
    params->warmup_iter     = 10000;
    params->am_hdr_size     = 8;
    params->alignment       = ucs_get_page_size();
    params->max_iter        = 1000000l;
    params->max_time        = 0.0;
    params->report_interval = 1.0;
    params->flags           = UCX_PERF_TEST_FLAG_VERBOSE;
    params->uct.fc_window   = UCT_PERF_TEST_MAX_FC_WINDOW;
    params->uct.data_layout = UCT_PERF_DATA_LAYOUT_SHORT;
    params->mem_type        = UCT_MD_MEM_TYPE_HOST;
    params->msg_size_cnt    = 1;
    params->iov_stride      = 0;
    params->ucp.send_datatype = UCP_PERF_DATATYPE_CONTIG;
    params->ucp.recv_datatype = UCP_PERF_DATATYPE_CONTIG;
    strcpy(params->uct.dev_name, "<none>");
    strcpy(params->uct.tl_name, "<none>");

    params->msg_size_list    = malloc(sizeof(*params->msg_size_list) *
                                      params->msg_size_cnt);
    params->msg_size_list[0] = 8;
}

static ucs_status_t parse_test_params(ucx_perf_params_t *params, char opt, const char *optarg)
{
    test_type_t *test;
    char *optarg2 = NULL;

    switch (opt) {
    case 'd':
        ucs_snprintf_zero(params->uct.dev_name, sizeof(params->uct.dev_name),
                          "%s", optarg);
        return UCS_OK;
    case 'x':
        ucs_snprintf_zero(params->uct.tl_name, sizeof(params->uct.tl_name),
                          "%s", optarg);
        return UCS_OK;
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
        if (!strcmp(optarg, "short")) {
            params->uct.data_layout   = UCT_PERF_DATA_LAYOUT_SHORT;
        } else if (!strcmp(optarg, "bcopy")) {
            params->uct.data_layout   = UCT_PERF_DATA_LAYOUT_BCOPY;
        } else if (!strcmp(optarg, "zcopy")) {
            params->uct.data_layout   = UCT_PERF_DATA_LAYOUT_ZCOPY;
        } else if (UCS_OK == parse_ucp_datatype_params(optarg,
                                                       &params->ucp.send_datatype)) {
            optarg2 = strchr(optarg, ',');
            if (optarg2) {
                if (UCS_OK != parse_ucp_datatype_params(optarg2 + 1,
                                                       &params->ucp.recv_datatype)) {
                    return -1;
                }
            }
        } else {
            ucs_error("Invalid option argument for -D");
            return -1;
        }
        return UCS_OK;
    case 'i':
        params->iov_stride = atol(optarg);
        return UCS_OK;
    case 'n':
        params->max_iter = atol(optarg);
        return UCS_OK;
    case 's':
        return parse_message_sizes_params(optarg, params);
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
    case 'B':
        params->flags |= UCX_PERF_TEST_FLAG_MAP_NONBLOCK;
        return UCS_OK;
    case 'q':
        params->flags &= ~UCX_PERF_TEST_FLAG_VERBOSE;
        return UCS_OK;
    case 'C':
        params->flags |= UCX_PERF_TEST_FLAG_TAG_WILDCARD;
        return UCS_OK;
    case 'U':
        params->flags |= UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE;
        return UCS_OK;
    case 'M':
        if (!strcmp(optarg, "single")) {
            params->thread_mode = UCS_THREAD_MODE_SINGLE;
            return UCS_OK;
        } else if (!strcmp(optarg, "serialized")) {
            params->thread_mode = UCS_THREAD_MODE_SERIALIZED;
            return UCS_OK;
        } else if (!strcmp(optarg, "multi")) {
            params->thread_mode = UCS_THREAD_MODE_MULTI;
            return UCS_OK;
        } else {
            ucs_error("Invalid option argument for -M");
            return UCS_ERR_INVALID_PARAM;
        }
    case 'T':
        params->thread_count = atoi(optarg);
        params->thread_mode = UCS_THREAD_MODE_MULTI;
        return UCS_OK;
    case 'A':
        if (!strcmp(optarg, "thread")) {
            params->async_mode = UCS_ASYNC_MODE_THREAD;
            return UCS_OK;
        } else if (!strcmp(optarg, "signal")) {
            params->async_mode = UCS_ASYNC_MODE_SIGNAL;
            return UCS_OK;
        } else {
            ucs_error("Invalid option argument for -A");
            return UCS_ERR_INVALID_PARAM;
        }
    case 'r':
        if (!strcmp(optarg, "recv_data")) {
            params->flags |= UCX_PERF_TEST_FLAG_STREAM_RECV_DATA;
            return UCS_OK;
        } else if (!strcmp(optarg, "recv")) {
            params->flags &= ~UCX_PERF_TEST_FLAG_STREAM_RECV_DATA;
            return UCS_OK;
        }
        return UCS_ERR_INVALID_PARAM;
    case 'm':
        if (!strcmp(optarg, "host")) {
            params->mem_type = UCT_MD_MEM_TYPE_HOST;
            return UCS_OK;
        } else if(!strcmp(optarg, "cuda")) {
#if HAVE_CUDA
            params->mem_type = UCT_MD_MEM_TYPE_CUDA;
            return UCS_OK;
#else
            ucs_error("not built with cuda support");
            return UCS_ERR_INVALID_PARAM;
#endif
        }
        return UCS_ERR_INVALID_PARAM;
    default:
       return UCS_ERR_INVALID_PARAM;
    }
}

static ucs_status_t read_batch_file(FILE *batch_file, const char *file_name,
                                    int *line_num, ucx_perf_params_t *params,
                                    char** test_name_p)
{
#define MAX_SIZE 256
#define MAX_ARG_SIZE 2048
    ucs_status_t status;
    char buf[MAX_ARG_SIZE];
    int argc;
    char *argv[MAX_SIZE + 1];
    int c;
    char *p;

    do {
        if (fgets(buf, sizeof(buf) - 1, batch_file) == NULL) {
            return UCS_ERR_NO_ELEM;
        }
        ++(*line_num);

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
            ucs_error("in batch file '%s' line %d: -%c %s: %s",
                      file_name, *line_num, c, optarg, ucs_status_string(status));
            return status;
        }
    }

    *test_name_p = strdup(argv[0]);
    return UCS_OK;
}

static ucs_status_t parse_opts(struct perftest_context *ctx, int mpi_initialized,
                               int argc, char **argv)
{
    ucs_status_t status;
    int c;

    ucs_trace_func("");

    init_test_params(&ctx->params);
    ctx->server_addr            = NULL;
    ctx->num_batch_files        = 0;
    ctx->port                   = 13337;
    ctx->flags                  = 0;
    ctx->mpi                    = mpi_initialized;

    optind = 1;
    while ((c = getopt (argc, argv, "p:b:Nfvc:P:h" TEST_PARAMS_ARGS)) != -1) {
        switch (c) {
        case 'p':
            ctx->port = atoi(optarg);
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
        case 'P':
#if HAVE_MPI
            ctx->mpi = atoi(optarg) && mpi_initialized;
            break;
#endif
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

static unsigned sock_rte_group_size(void *rte_group)
{
    return 2;
}

static unsigned sock_rte_group_index(void *rte_group)
{
    sock_rte_group_t *group = rte_group;
    return group->is_server ? 0 : 1;
}

static void sock_rte_barrier(void *rte_group, void (*progress)(void *arg),
                             void *arg)
{
#pragma omp master
  {
    sock_rte_group_t *group = rte_group;
    const unsigned magic = 0xdeadbeef;
    unsigned sync;

    sync = magic;
    safe_send(group->connfd, &sync, sizeof(unsigned), progress, arg);

    sync = 0;
    safe_recv(group->connfd, &sync, sizeof(unsigned), progress, arg);

    ucs_assert(sync == magic);
  }
#pragma omp barrier
}

static void sock_rte_post_vec(void *rte_group, const struct iovec *iovec,
                              int iovcnt, void **req)
{
    sock_rte_group_t *group = rte_group;
    size_t size;
    int i;

    size = 0;
    for (i = 0; i < iovcnt; ++i) {
        size += iovec[i].iov_len;
    }

    safe_send(group->connfd, &size, sizeof(size), NULL, NULL);
    for (i = 0; i < iovcnt; ++i) {
        safe_send(group->connfd, iovec[i].iov_base, iovec[i].iov_len, NULL,
                  NULL);
    }
}

static void sock_rte_recv(void *rte_group, unsigned src, void *buffer,
                          size_t max, void *req)
{
    sock_rte_group_t *group = rte_group;
    int group_index;
    size_t size;

    group_index = sock_rte_group_index(rte_group);
    if (src == group_index) {
        return;
    }

    ucs_assert_always(src == (1 - group_index));
    safe_recv(group->connfd, &size, sizeof(size), NULL, NULL);
    ucs_assert_always(size <= max);
    safe_recv(group->connfd, buffer, size, NULL, NULL);
}

static void sock_rte_report(void *rte_group, const ucx_perf_result_t *result,
                            void *arg, int is_final)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final);
}

static ucx_perf_rte_t sock_rte = {
    .group_size    = sock_rte_group_size,
    .group_index   = sock_rte_group_index,
    .barrier       = sock_rte_barrier,
    .post_vec      = sock_rte_post_vec,
    .recv          = sock_rte_recv,
    .exchange_vec  = (void*)ucs_empty_function,
    .report        = sock_rte_report,
};

static ucs_status_t setup_sock_rte(struct perftest_context *ctx)
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
        safe_recv(connfd, &ctx->params, sizeof(ctx->params), NULL, NULL);
        if (ctx->params.msg_size_cnt) {
            ctx->params.msg_size_list = malloc(sizeof(*ctx->params.msg_size_list) *
                                               ctx->params.msg_size_cnt);
            if (NULL == ctx->params.msg_size_list) {
                status = UCS_ERR_NO_MEMORY;
                goto err_close_connfd;
            }
            safe_recv(connfd, ctx->params.msg_size_list,
                      sizeof(*ctx->params.msg_size_list) * ctx->params.msg_size_cnt,
                      NULL, NULL);
        }

        ctx->sock_rte_group.connfd    = connfd;
        ctx->sock_rte_group.is_server = 1;
    } else {
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

        safe_send(sockfd, &ctx->params, sizeof(ctx->params), NULL, NULL);
        if (ctx->params.msg_size_cnt) {
            safe_send(sockfd, ctx->params.msg_size_list,
                      sizeof(*ctx->params.msg_size_list) * ctx->params.msg_size_cnt,
                      NULL, NULL);
        }

        ctx->sock_rte_group.connfd    = sockfd;
        ctx->sock_rte_group.is_server = 0;
    }

    if (ctx->sock_rte_group.is_server) {
        ctx->flags |= TEST_FLAG_PRINT_TEST;
    } else {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.rte_group         = &ctx->sock_rte_group;
    ctx->params.rte               = &sock_rte;
    ctx->params.report_arg        = ctx;
    return UCS_OK;

err_close_connfd:
    close(connfd);
    goto err;
err_close_sockfd:
    close(sockfd);
err:
    return status;
}

static ucs_status_t cleanup_sock_rte(struct perftest_context *ctx)
{
    close(ctx->sock_rte_group.connfd);
    return UCS_OK;
}

#if HAVE_MPI
static unsigned mpi_rte_group_size(void *rte_group)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

static unsigned mpi_rte_group_index(void *rte_group)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

static void mpi_rte_barrier(void *rte_group, void (*progress)(void *arg),
                            void *arg)
{
    int group_size, my_rank, i;
    MPI_Request *reqs;
    int nreqs = 0;
    int dummy;
    int flag;

#pragma omp master

    /*
     * Naive non-blocking barrier implementation over send/recv, to call user
     * progress while waiting for completion.
     * Not using MPI_Ibarrier to be compatible with MPI-1.
     */

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    /* allocate maximal possible number of requests */
    reqs = (MPI_Request*)alloca(sizeof(*reqs) * group_size);

    if (my_rank == 0) {
        /* root gathers "ping" from all other ranks */
        for (i = 1; i < group_size; ++i) {
            MPI_Irecv(&dummy, 0, MPI_INT,
                      i /* source */,
                      1 /* tag */,
                      MPI_COMM_WORLD,
                      &reqs[nreqs++]);
        }
    } else {
        /* every non-root rank sends "ping" and waits for "pong" */
        MPI_Send(&dummy, 0, MPI_INT,
                 0 /* dest */,
                 1 /* tag */,
                 MPI_COMM_WORLD);
        MPI_Irecv(&dummy, 0, MPI_INT,
                  0 /* source */,
                  2 /* tag */,
                  MPI_COMM_WORLD,
                  &reqs[nreqs++]);
    }

    /* Waiting for receive requests */
    do {
        MPI_Testall(nreqs, reqs, &flag, MPI_STATUSES_IGNORE);
        progress(arg);
    } while (!flag);

    if (my_rank == 0) {
        /* root sends "pong" to all ranks */
        for (i = 1; i < group_size; ++i) {
            MPI_Send(&dummy, 0, MPI_INT,
                     i /* dest */,
                     2 /* tag */,
                     MPI_COMM_WORLD);
       }
    }

#pragma omp barrier
}

static void mpi_rte_post_vec(void *rte_group, const struct iovec *iovec,
                             int iovcnt, void **req)
{
    int group_size;
    int my_rank;
    int dest, i;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    for (dest = 0; dest < group_size; ++dest) {
        if (dest == my_rank) {
            continue;
        }

        for (i = 0; i < iovcnt; ++i) {
            MPI_Send(iovec[i].iov_base, iovec[i].iov_len, MPI_BYTE, dest,
                     i == (iovcnt - 1), /* Send last iov with tag == 1 */
                     MPI_COMM_WORLD);
        }
    }

    *req = (void*)(uintptr_t)1;
}

static void mpi_rte_recv(void *rte_group, unsigned src, void *buffer, size_t max,
                         void *req)
{
    MPI_Status status;
    size_t offset;
    int my_rank;
    int count;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (src == my_rank) {
        return;
    }

    offset = 0;
    do {
        ucs_assert_always(offset < max);
        MPI_Recv(buffer + offset, max - offset, MPI_BYTE, src, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &count);
        offset += count;
    } while (status.MPI_TAG != 1);
}

static void mpi_rte_report(void *rte_group, const ucx_perf_result_t *result,
                           void *arg, int is_final)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final);
}

static ucx_perf_rte_t mpi_rte = {
    .group_size    = mpi_rte_group_size,
    .group_index   = mpi_rte_group_index,
    .barrier       = mpi_rte_barrier,
    .post_vec      = mpi_rte_post_vec,
    .recv          = mpi_rte_recv,
    .exchange_vec  = (void*)ucs_empty_function,
    .report        = mpi_rte_report,
};
#elif HAVE_RTE
static unsigned ext_rte_group_size(void *rte_group)
{
    rte_group_t group = (rte_group_t)rte_group;
    return rte_group_size(group);
}

static unsigned ext_rte_group_index(void *rte_group)
{
    rte_group_t group = (rte_group_t)rte_group;
    return rte_group_rank(group);
}

static void ext_rte_barrier(void *rte_group, void (*progress)(void *arg),
                            void *arg)
{
#pragma omp master
  {
    rte_group_t group = (rte_group_t)rte_group;
    int rc;

    rc = rte_barrier(group);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_barrier");
    }
  }
#pragma omp barrier
}

static void ext_rte_post_vec(void *rte_group, const struct iovec* iovec,
                             int iovcnt, void **req)
{
    rte_group_t group = (rte_group_t)rte_group;
    rte_srs_session_t session;
    rte_iovec_t *r_vec;
    int i, rc;

    rc = rte_srs_session_create(group, 0, &session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_session_create");
    }

    r_vec = calloc(iovcnt, sizeof(rte_iovec_t));
    if (r_vec == NULL) {
        return;
    }
    for (i = 0; i < iovcnt; ++i) {
        r_vec[i].iov_base = iovec[i].iov_base;
        r_vec[i].type     = rte_datatype_uint8_t;
        r_vec[i].count    = iovec[i].iov_len;
    }
    rc = rte_srs_set_data(session, "KEY_PERF", r_vec, iovcnt);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_set_data");
    }
    *req = session;
    free(r_vec);
}

static void ext_rte_recv(void *rte_group, unsigned src, void *buffer,
                         size_t max, void *req)
{
    rte_group_t group         = (rte_group_t)rte_group;
    rte_srs_session_t session = (rte_srs_session_t)req;
    void *rte_buffer = NULL;
    rte_iovec_t r_vec;
    uint32_t offset;
    int size;
    int rc;

    rc = rte_srs_get_data(session, rte_group_index_to_ec(group, src),
                          "KEY_PERF", &rte_buffer, &size);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_get_data");
        return;
    }

    r_vec.iov_base = buffer;
    r_vec.type     = rte_datatype_uint8_t;
    r_vec.count    = max;

    offset = 0;
    rte_unpack(&r_vec, rte_buffer, &offset);

    rc = rte_srs_session_destroy(session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_session_destroy");
    }
    free(rte_buffer);
}

static void ext_rte_exchange_vec(void *rte_group, void * req)
{
    rte_srs_session_t session = (rte_srs_session_t)req;
    int rc;

    rc = rte_srs_exchange_data(session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_exchange_data");
    }
}

static void ext_rte_report(void *rte_group, const ucx_perf_result_t *result,
                           void *arg, int is_final)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final);
}

static ucx_perf_rte_t ext_rte = {
    .group_size    = ext_rte_group_size,
    .group_index   = ext_rte_group_index,
    .barrier       = ext_rte_barrier,
    .report        = ext_rte_report,
    .post_vec      = ext_rte_post_vec,
    .recv          = ext_rte_recv,
    .exchange_vec  = ext_rte_exchange_vec,
};
#endif

static ucs_status_t setup_mpi_rte(struct perftest_context *ctx)
{
    ucs_trace_func("");

#if HAVE_MPI
    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        ucs_error("This test should run with exactly 2 processes (actual: %d)", size);
        return UCS_ERR_INVALID_PARAM;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 1) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.rte_group         = NULL;
    ctx->params.rte               = &mpi_rte;
    ctx->params.report_arg        = ctx;
#elif HAVE_RTE
    rte_group_t group;

    rte_init(NULL, NULL, &group);
    if (1 == rte_group_rank(group)) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.rte_group         = group;
    ctx->params.rte               = &ext_rte;
    ctx->params.report_arg        = ctx;
#endif
    return UCS_OK;
}

static ucs_status_t cleanup_mpi_rte(struct perftest_context *ctx)
{
#if HAVE_RTE
    rte_finalize();
#endif
    return UCS_OK;
}

static ucs_status_t check_system(struct perftest_context *ctx)
{
    cpu_set_t cpuset;
    unsigned i, count, nr_cpus;
    int ret;

    ucs_trace_func("");

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
    int line_num;

    ucs_trace_func("depth=%u, num_files=%u", depth, ctx->num_batch_files);

    if (depth >= ctx->num_batch_files) {
        print_test_name(ctx);
        return ucx_perf_run(parent_params, &result);
    }

    batch_file = fopen(ctx->batch_files[depth], "r");
    if (batch_file == NULL) {
        ucs_error("Failed to open batch file '%s': %m", ctx->batch_files[depth]);
        return UCS_ERR_IO_ERROR;
    }

    params   = *parent_params;
    line_num = 0;
    while ((status = read_batch_file(batch_file, ctx->batch_files[depth],
                                     &line_num, &params,
                                     &ctx->test_names[depth])) == UCS_OK) {
        status = run_test_recurs(ctx, &params, depth + 1);
        free(ctx->test_names[depth]);
        if ((NULL == parent_params->msg_size_list) &&
            (NULL != params.msg_size_list)) {
            free(params.msg_size_list);
            params.msg_size_list = NULL;
        }
        params = *parent_params;
    }

    fclose(batch_file);
    return UCS_OK;
}

static ucs_status_t run_test(struct perftest_context *ctx)
{
    ucs_status_t status;

    ucs_trace_func("");

    setlocale(LC_ALL, "en_US");

    print_header(ctx);

    status = run_test_recurs(ctx, &ctx->params, 0);
    if (status != UCS_OK) {
        ucs_error("Failed to run test: %s", ucs_status_string(status));
    }

    return status;
}

int main(int argc, char **argv)
{
    struct perftest_context ctx;
    ucs_status_t status;
    int mpi_initialized;
    int mpi_rte;
    int ret;

#if HAVE_MPI
    mpi_initialized = !isatty(0) && (MPI_Init(&argc, &argv) == 0);
#else
    mpi_initialized = 0;
#endif

    /* Parse command line */
    status = parse_opts(&ctx, mpi_initialized, argc, argv);
    if (status != UCS_OK) {
        ret = (status == UCS_ERR_CANCELED) ? 0 : -127;
        goto out;
    }

#ifdef __COVERITY__
    /* coverity[dont_call] */
    mpi_rte = rand(); /* Shut up deadcode error */
#endif

    if (ctx.mpi) {
        mpi_rte = 1;
    } else {
#if HAVE_RTE
        mpi_rte = 1;
#else
        mpi_rte = 0;
#endif
    }

    status = check_system(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out;
    }

    /* Create RTE */
    status = (mpi_rte) ? setup_mpi_rte(&ctx) : setup_sock_rte(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out;
    }

    /* Run the test */
    status = run_test(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_cleanup_rte;
    }

    ret = 0;

out_cleanup_rte:
    (mpi_rte) ? cleanup_mpi_rte(&ctx) : cleanup_sock_rte(&ctx);
out:
    if (ctx.params.msg_size_list) {
        free(ctx.params.msg_size_list);
    }
    if (mpi_initialized) {
#if HAVE_MPI
        MPI_Finalize();
#endif
    }
    return ret;
}
