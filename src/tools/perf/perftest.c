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

#include "api/libperf.h"
#include "lib/libperf_int.h"

#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/sock.h>
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
#if defined (HAVE_MPI)
#  include <mpi.h>
#elif defined (HAVE_RTE)
#   include<rte.h>
#endif

#define MAX_BATCH_FILES         32
#define MAX_CPUS                1024
#define TL_RESOURCE_NAME_NONE   "<none>"
#define TEST_PARAMS_ARGS        "t:n:s:W:O:w:D:i:H:oSCIqM:r:T:d:x:A:BUm:"
#define TEST_ID_UNDEFINED       -1

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
    const char                   *overhead_lat;
    unsigned                     window_size;
} test_type_t;


typedef struct perftest_params {
    ucx_perf_params_t            super;
    int                          test_id;
} perftest_params_t;


struct perftest_context {
    perftest_params_t            params;
    const char                   *server_addr;
    int                          port;
    int                          mpi;
    unsigned                     num_cpus;
    unsigned                     cpus[MAX_CPUS];
    unsigned                     flags;

    unsigned                     num_batch_files;
    char                         *batch_files[MAX_BATCH_FILES];
    char                         *test_names[MAX_BATCH_FILES];

    sock_rte_group_t             sock_rte_group;
};


test_type_t tests[] = {
    {"am_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
     "active message latency", "latency", 1},

    {"put_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
     "put latency", "latency", 1},

    {"add_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_PINGPONG,
     "atomic add latency", "latency", 1},

    {"get", UCX_PERF_API_UCT, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "get latency / bandwidth / message rate", "latency", 1},

    {"fadd", UCX_PERF_API_UCT, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic fetch-and-add latency / rate", "latency", 1},

    {"swap", UCX_PERF_API_UCT, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic swap latency / rate", "latency", 1},

    {"cswap", UCX_PERF_API_UCT, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic compare-and-swap latency / rate", "latency", 1},

    {"am_bw", UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "active message bandwidth / message rate", "overhead", 1},

    {"put_bw", UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "put bandwidth / message rate", "overhead", 1},

    {"add_mr", UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic add message rate", "overhead", 1},

    {"tag_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
     "tag match latency", "latency", 1},

    {"tag_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "tag match bandwidth", "overhead", 32},

    {"tag_sync_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_PINGPONG,
     "tag sync match latency", "latency", 1},

    {"tag_sync_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "tag sync match bandwidth", "overhead", 32},

    {"ucp_put_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
     "put latency", "latency", 1},

    {"ucp_put_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "put bandwidth", "overhead", 32},

    {"ucp_get", UCX_PERF_API_UCP, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "get latency / bandwidth / message rate", "latency", 1},

    {"ucp_add", UCX_PERF_API_UCP, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic add bandwidth / message rate", "overhead", 1},

    {"ucp_fadd", UCX_PERF_API_UCP, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic fetch-and-add latency / bandwidth / rate", "latency", 1},

    {"ucp_swap", UCX_PERF_API_UCP, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic swap latency / bandwidth / rate", "latency", 1},

    {"ucp_cswap", UCX_PERF_API_UCP, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic compare-and-swap latency / bandwidth / rate", "latency", 1},

    {"stream_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "stream bandwidth", "overhead", 1},

    {"stream_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_PINGPONG,
     "stream latency", "latency", 1},

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
    typedef ssize_t (*sock_call)(int, void *, size_t, int);

    return sock_io(sock, (sock_call)send, POLLOUT, data, size, progress, arg, "send");
}

static int safe_recv(int sock, void *data, size_t size,
                     void (*progress)(void *arg), void *arg)
{
    return sock_io(sock, recv, POLLIN, data, size, progress, arg, "recv");
}

static void print_progress(char **test_names, unsigned num_names,
                           const ucx_perf_result_t *result, unsigned flags,
                           int final, int is_server, int is_multi_thread)
{
    static const char *fmt_csv;
    static const char *fmt_numeric;
    static const char *fmt_plain;
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

#if _OPENMP
    if (!final) {
        printf("[thread %d]", omp_get_thread_num());
    } else if (flags & TEST_FLAG_PRINT_RESULTS) {
        printf("Final:    ");
    }
#endif

    if (is_multi_thread && final) {
        fmt_csv     = "%4.0f,%.3f,%.2f,%.0f\n";
        fmt_numeric = "%'18.0f %29.3f %22.2f %'24.0f\n";
        fmt_plain   = "%18.0f %29.3f %22.2f %23.0f\n";

        printf((flags & TEST_FLAG_PRINT_CSV)   ? fmt_csv :
               (flags & TEST_FLAG_NUMERIC_FMT) ? fmt_numeric :
                                                 fmt_plain,
               (double)result->iters,
               result->latency.total_average * 1000000.0,
               result->bandwidth.total_average / (1024.0 * 1024.0),
               result->msgrate.total_average);
    } else {
        fmt_csv     = "%4.0f,%.3f,%.3f,%.3f,%.2f,%.2f,%.0f,%.0f\n";
        fmt_numeric = "%'18.0f %9.3f %9.3f %9.3f %11.2f %10.2f %'11.0f %'11.0f\n";
        fmt_plain   = "%18.0f %9.3f %9.3f %9.3f %11.2f %10.2f %11.0f %11.0f\n";

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
    }

    fflush(stdout);
}

static void print_header(struct perftest_context *ctx)
{
    const char *overhead_lat_str;
    const char *test_data_str;
    const char *test_api_str;
    test_type_t *test;
    unsigned i;

    test = (ctx->params.test_id == TEST_ID_UNDEFINED) ? NULL :
           &tests[ctx->params.test_id];

    if ((ctx->flags & TEST_FLAG_PRINT_TEST) && (test != NULL)) {
        if (test->api == UCX_PERF_API_UCT) {
            test_api_str = "transport layer";
            switch (ctx->params.super.uct.data_layout) {
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
            test_api_str  = "protocol layer";
            test_data_str = "(automatic)"; /* TODO contig/stride/stream */
        } else {
            return;
        }

        printf("+------------------------------------------------------------------------------------------+\n");
        printf("| API:          %-60s               |\n", test_api_str);
        printf("| Test:         %-60s               |\n", test->desc);
        printf("| Data layout:  %-60s               |\n", test_data_str);
        printf("| Send memory:  %-60s               |\n", ucs_memory_type_names[ctx->params.super.send_mem_type]);
        printf("| Recv memory:  %-60s               |\n", ucs_memory_type_names[ctx->params.super.recv_mem_type]);
        printf("| Message size: %-60zu               |\n", ucx_perf_get_message_size(&ctx->params.super));
    }

    if (ctx->flags & TEST_FLAG_PRINT_CSV) {
        if (ctx->flags & TEST_FLAG_PRINT_RESULTS) {
            for (i = 0; i < ctx->num_batch_files; ++i) {
                printf("%s,", ucs_basename(ctx->batch_files[i]));
            }
            printf("iterations,typical_lat,avg_lat,overall_lat,avg_bw,overall_bw,avg_mr,overall_mr\n");
        }
    } else {
        if (ctx->flags & TEST_FLAG_PRINT_RESULTS) {
            overhead_lat_str = (test == NULL) ? "overhead" : test->overhead_lat;

            printf("+--------------+--------------+-----------------------------+---------------------+-----------------------+\n");
            printf("|              |              |      %8s (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |\n", overhead_lat_str);
            printf("+--------------+--------------+---------+---------+---------+----------+----------+-----------+-----------+\n");
            printf("|    Stage     | # iterations | typical | average | overall |  average |  overall |  average  |  overall  |\n");
            printf("+--------------+--------------+---------+---------+---------+----------+----------+-----------+-----------+\n");
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
        strcpy(buf, "+--------------+--------------+---------+---------+---------+----------+----------+-----------+-----------+");

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

static void print_memory_type_usage(void)
{
    ucs_memory_type_t it;
    for (it = UCS_MEMORY_TYPE_HOST; it < UCS_MEMORY_TYPE_LAST; it++) {
        if (ucx_perf_mem_type_allocators[it] != NULL) {
            printf("                        %s - %s\n",
                   ucs_memory_type_names[it],
                   ucs_memory_type_descs[it]);
        }
    }
}

static void usage(const struct perftest_context *ctx, const char *program)
{
    static const char* api_names[] = {
        [UCX_PERF_API_UCT] = "UCT",
        [UCX_PERF_API_UCP] = "UCP"
    };
    test_type_t *test;
    int UCS_V_UNUSED rank;

#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ctx->mpi && (rank != 0)) {
        return;
    }
#endif

#if defined (HAVE_MPI)
    printf("  Note: test can be also launched as an MPI application\n");
    printf("\n");
#elif defined (HAVE_RTE)
    printf("  Note: this test can be also launched as an libRTE application\n");
    printf("\n");
#endif
    printf("  Usage: %s [ server-hostname ] [ options ]\n", program);
    printf("\n");
    printf("  Common options:\n");
    printf("     -t <test>      test to run:\n");
    for (test = tests; test->name; ++test) {
        printf("    %13s - %s %s\n", test->name,
               api_names[test->api], test->desc);
    }
    printf("\n");
    printf("     -s <size>      list of scatter-gather sizes for single message (%zu)\n",
                                ctx->params.super.msg_size_list[0]);
    printf("                    for example: \"-s 16,48,8192,8192,14\"\n");
    printf("     -m <send mem type>[,<recv mem type>]\n");
    printf("                    memory type of message for sender and receiver (host)\n");
    print_memory_type_usage();
    printf("     -n <iters>     number of iterations to run (%"PRIu64")\n", ctx->params.super.max_iter);
    printf("     -w <iters>     number of warm-up iterations (%"PRIu64")\n",
                                ctx->params.super.warmup_iter);
    printf("     -c <cpulist>   set affinity to this CPU list (separated by comma) (off)\n");
    printf("     -O <count>     maximal number of uncompleted outstanding sends\n");
    printf("     -i <offset>    distance between consecutive scatter-gather entries (%zu)\n",
                                ctx->params.super.iov_stride);
    printf("     -T <threads>   number of threads in the test (%d)\n",
                                ctx->params.super.thread_count);
    printf("     -o             do not progress the responder in one-sided tests\n");
    printf("     -B             register memory with NONBLOCK flag\n");
    printf("     -b <file>      read and execute tests from a batch file: every line in the\n");
    printf("                    file is a test to run, first word is test name, the rest of\n");
    printf("                    the line is command-line arguments for the test.\n");
    printf("     -p <port>      TCP port to use for data exchange (%d)\n", ctx->port);
#ifdef HAVE_MPI
    printf("     -P <0|1>       disable/enable MPI mode (%d)\n", ctx->mpi);
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
    printf("                        short - short messages (default, cannot be used for get)\n");
    printf("                        bcopy - copy-out (cannot be used for atomics)\n");
    printf("                        zcopy - zero-copy (cannot be used for atomics)\n");
    printf("                        iov    - scatter-gather list (iovec)\n");
    printf("     -W <count>     flow control window size, for active messages (%u)\n",
                                ctx->params.super.uct.fc_window);
    printf("     -H <size>      active message header size (%zu)\n",
                                ctx->params.super.am_hdr_size);
    printf("     -A <mode>      asynchronous progress mode (thread_spinlock)\n");
    printf("                        thread_spinlock - separate progress thread with spin locking\n");
    printf("                        thread_mutex - separate progress thread with mutex locking\n");
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
    printf("     -I             create context with wakeup feature enabled\n");
    printf("\n");
    printf("   NOTE: When running UCP tests, transport and device should be specified by\n");
    printf("         environment variables: UCX_TLS and UCX_[SELF|SHM|NET]_DEVICES.\n");
    printf("\n");
}

static ucs_status_t parse_ucp_datatype_params(const char *opt_arg,
                                              ucp_perf_datatype_t *datatype)
{
    const char  *iov_type         = "iov";
    const size_t iov_type_size    = strlen("iov");
    const char  *contig_type      = "contig";
    const size_t contig_type_size = strlen("contig");

    if (0 == strncmp(opt_arg, iov_type, iov_type_size)) {
        *datatype = UCP_PERF_DATATYPE_IOV;
    } else if (0 == strncmp(opt_arg, contig_type, contig_type_size)) {
        *datatype = UCP_PERF_DATATYPE_CONTIG;
    } else {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t parse_mem_type(const char *opt_arg,
                                   ucs_memory_type_t *mem_type)
{
    ucs_memory_type_t it;
    for (it = UCS_MEMORY_TYPE_HOST; it < UCS_MEMORY_TYPE_LAST; it++) {
        if(!strcmp(opt_arg, ucs_memory_type_names[it]) &&
           (ucx_perf_mem_type_allocators[it] != NULL)) {
            *mem_type = it;
            return UCS_OK;
        }
    }
    ucs_error("Unsupported memory type: \"%s\"", opt_arg);
    return UCS_ERR_INVALID_PARAM;
}

static ucs_status_t parse_mem_type_params(const char *opt_arg,
                                          ucs_memory_type_t *send_mem_type,
                                          ucs_memory_type_t *recv_mem_type)
{
    const char *delim = ",";
    char *token       = strtok((char*)opt_arg, delim);

    if (UCS_OK != parse_mem_type(token, send_mem_type)) {
        return UCS_ERR_INVALID_PARAM;
    }

    token = strtok(NULL, delim);
    if (NULL == token) {
        *recv_mem_type = *send_mem_type;
        return UCS_OK;
    } else {
        return parse_mem_type(token, recv_mem_type);
    }
}

static ucs_status_t parse_message_sizes_params(const char *opt_arg,
                                               ucx_perf_params_t *params)
{
    const char delim = ',';
    size_t *msg_size_list, token_num, token_it;
    char *optarg_ptr, *optarg_ptr2;

    optarg_ptr = (char *)opt_arg;
    token_num  = 0;
    /* count the number of given message sizes */
    while ((optarg_ptr = strchr(optarg_ptr, delim)) != NULL) {
        ++optarg_ptr;
        ++token_num;
    }
    ++token_num;

    msg_size_list = realloc(params->msg_size_list,
                            sizeof(*params->msg_size_list) * token_num);
    if (NULL == msg_size_list) {
        return UCS_ERR_NO_MEMORY;
    }

    params->msg_size_list = msg_size_list;

    optarg_ptr = (char *)opt_arg;
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

static ucs_status_t init_test_params(perftest_params_t *params)
{
    memset(params, 0, sizeof(*params));
    params->super.api               = UCX_PERF_API_LAST;
    params->super.command           = UCX_PERF_CMD_LAST;
    params->super.test_type         = UCX_PERF_TEST_TYPE_LAST;
    params->super.thread_mode       = UCS_THREAD_MODE_SINGLE;
    params->super.thread_count      = 1;
    params->super.async_mode        = UCS_ASYNC_THREAD_LOCK_TYPE;
    params->super.wait_mode         = UCX_PERF_WAIT_MODE_LAST;
    params->super.max_outstanding   = 0;
    params->super.warmup_iter       = 10000;
    params->super.am_hdr_size       = 8;
    params->super.alignment         = ucs_get_page_size();
    params->super.max_iter          = 1000000l;
    params->super.max_time          = 0.0;
    params->super.report_interval   = 1.0;
    params->super.flags             = UCX_PERF_TEST_FLAG_VERBOSE;
    params->super.uct.fc_window     = UCT_PERF_TEST_MAX_FC_WINDOW;
    params->super.uct.data_layout   = UCT_PERF_DATA_LAYOUT_SHORT;
    params->super.send_mem_type     = UCS_MEMORY_TYPE_HOST;
    params->super.recv_mem_type     = UCS_MEMORY_TYPE_HOST;
    params->super.msg_size_cnt      = 1;
    params->super.iov_stride        = 0;
    params->super.ucp.send_datatype = UCP_PERF_DATATYPE_CONTIG;
    params->super.ucp.recv_datatype = UCP_PERF_DATATYPE_CONTIG;
    strcpy(params->super.uct.dev_name, TL_RESOURCE_NAME_NONE);
    strcpy(params->super.uct.tl_name,  TL_RESOURCE_NAME_NONE);

    params->super.msg_size_list = calloc(params->super.msg_size_cnt,
                                         sizeof(*params->super.msg_size_list));
    if (params->super.msg_size_list == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    params->super.msg_size_list[0] = 8;
    params->test_id                = TEST_ID_UNDEFINED;

    return UCS_OK;
}

static ucs_status_t parse_test_params(perftest_params_t *params, char opt,
                                      const char *opt_arg)
{
    char *optarg2 = NULL;
    test_type_t *test;
    unsigned i;

    switch (opt) {
    case 'd':
        ucs_snprintf_zero(params->super.uct.dev_name,
                          sizeof(params->super.uct.dev_name), "%s", opt_arg);
        return UCS_OK;
    case 'x':
        ucs_snprintf_zero(params->super.uct.tl_name,
                          sizeof(params->super.uct.tl_name), "%s", opt_arg);
        return UCS_OK;
    case 't':
        for (i = 0; tests[i].name != NULL; ++i) {
            test = &tests[i];
            if (!strcmp(opt_arg, test->name)) {
                params->super.api       = test->api;
                params->super.command   = test->command;
                params->super.test_type = test->test_type;
                params->test_id         = i;
                break;
            }
        }
        if (params->test_id == TEST_ID_UNDEFINED) {
            ucs_error("Invalid option argument for -t");
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;
    case 'D':
        if (!strcmp(opt_arg, "short")) {
            params->super.uct.data_layout   = UCT_PERF_DATA_LAYOUT_SHORT;
        } else if (!strcmp(opt_arg, "bcopy")) {
            params->super.uct.data_layout   = UCT_PERF_DATA_LAYOUT_BCOPY;
        } else if (!strcmp(opt_arg, "zcopy")) {
            params->super.uct.data_layout   = UCT_PERF_DATA_LAYOUT_ZCOPY;
        } else if (UCS_OK == parse_ucp_datatype_params(opt_arg,
                                                       &params->super.ucp.send_datatype)) {
            optarg2 = strchr(opt_arg, ',');
            if (optarg2) {
                if (UCS_OK != parse_ucp_datatype_params(optarg2 + 1,
                                                       &params->super.ucp.recv_datatype)) {
                    return UCS_ERR_INVALID_PARAM;
                }
            }
        } else {
            ucs_error("Invalid option argument for -D");
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;
    case 'i':
        params->super.iov_stride = atol(opt_arg);
        return UCS_OK;
    case 'n':
        params->super.max_iter = atol(opt_arg);
        return UCS_OK;
    case 's':
        return parse_message_sizes_params(opt_arg, &params->super);
    case 'H':
        params->super.am_hdr_size = atol(opt_arg);
        return UCS_OK;
    case 'W':
        params->super.uct.fc_window = atoi(opt_arg);
        return UCS_OK;
    case 'O':
        params->super.max_outstanding = atoi(opt_arg);
        return UCS_OK;
    case 'w':
        params->super.warmup_iter = atol(opt_arg);
        return UCS_OK;
    case 'o':
        params->super.flags |= UCX_PERF_TEST_FLAG_ONE_SIDED;
        return UCS_OK;
    case 'B':
        params->super.flags |= UCX_PERF_TEST_FLAG_MAP_NONBLOCK;
        return UCS_OK;
    case 'q':
        params->super.flags &= ~UCX_PERF_TEST_FLAG_VERBOSE;
        return UCS_OK;
    case 'C':
        params->super.flags |= UCX_PERF_TEST_FLAG_TAG_WILDCARD;
        return UCS_OK;
    case 'U':
        params->super.flags |= UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE;
        return UCS_OK;
    case 'I':
        params->super.flags |= UCX_PERF_TEST_FLAG_WAKEUP;
        return UCS_OK;
    case 'M':
        if (!strcmp(opt_arg, "single")) {
            params->super.thread_mode = UCS_THREAD_MODE_SINGLE;
            return UCS_OK;
        } else if (!strcmp(opt_arg, "serialized")) {
            params->super.thread_mode = UCS_THREAD_MODE_SERIALIZED;
            return UCS_OK;
        } else if (!strcmp(opt_arg, "multi")) {
            params->super.thread_mode = UCS_THREAD_MODE_MULTI;
            return UCS_OK;
        } else {
            ucs_error("Invalid option argument for -M");
            return UCS_ERR_INVALID_PARAM;
        }
    case 'T':
        params->super.thread_count = atoi(opt_arg);
        return UCS_OK;
    case 'A':
        if (!strcmp(opt_arg, "thread") || !strcmp(opt_arg, "thread_spinlock")) {
            params->super.async_mode = UCS_ASYNC_MODE_THREAD_SPINLOCK;
            return UCS_OK;
        } else if (!strcmp(opt_arg, "thread_mutex")) {
            params->super.async_mode = UCS_ASYNC_MODE_THREAD_MUTEX;
            return UCS_OK;
        } else if (!strcmp(opt_arg, "signal")) {
            params->super.async_mode = UCS_ASYNC_MODE_SIGNAL;
            return UCS_OK;
        } else {
            ucs_error("Invalid option argument for -A");
            return UCS_ERR_INVALID_PARAM;
        }
    case 'r':
        if (!strcmp(opt_arg, "recv_data")) {
            params->super.flags |= UCX_PERF_TEST_FLAG_STREAM_RECV_DATA;
            return UCS_OK;
        } else if (!strcmp(opt_arg, "recv")) {
            params->super.flags &= ~UCX_PERF_TEST_FLAG_STREAM_RECV_DATA;
            return UCS_OK;
        }
        return UCS_ERR_INVALID_PARAM;
    case 'm':
        if (UCS_OK != parse_mem_type_params(opt_arg,
                                            &params->super.send_mem_type,
                                            &params->super.recv_mem_type)) {
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;
    default:
       return UCS_ERR_INVALID_PARAM;
    }
}

static ucs_status_t adjust_test_params(perftest_params_t *params,
                                       const char *error_prefix)
{
    test_type_t *test;

    if (params->test_id == TEST_ID_UNDEFINED) {
        ucs_error("%smissing test name", error_prefix);
        return UCS_ERR_INVALID_PARAM;
    }

    test = &tests[params->test_id];

    if (params->super.max_outstanding == 0) {
        params->super.max_outstanding = test->window_size;
    }

    return UCS_OK;
}

static ucs_status_t read_batch_file(FILE *batch_file, const char *file_name,
                                    int *line_num, perftest_params_t *params,
                                    char** test_name_p)
{
#define MAX_SIZE 256
#define MAX_ARG_SIZE 2048
    ucs_status_t status;
    char buf[MAX_ARG_SIZE];
    char error_prefix[MAX_ARG_SIZE];
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

    ucs_snprintf_safe(error_prefix, sizeof(error_prefix),
                      "in batch file '%s' line %d: ", file_name, *line_num);

    optind = 1;
    while ((c = getopt (argc, argv, TEST_PARAMS_ARGS)) != -1) {
        status = parse_test_params(params, c, optarg);
        if (status != UCS_OK) {
            ucs_error("%s-%c %s: %s", error_prefix, c, optarg,
                      ucs_status_string(status));
            return status;
        }
    }

    status = adjust_test_params(params, error_prefix);
    if (status != UCS_OK) {
        return status;
    }

    *test_name_p = strdup(argv[0]);
    return UCS_OK;
}

static ucs_status_t parse_cpus(char *opt_arg, struct perftest_context *ctx)
{
    char *endptr, *cpu_list = opt_arg;
    int cpu;

    ctx->num_cpus = 0;
    cpu           = strtol(cpu_list, &endptr, 10);

    while (((*endptr == ',') || (*endptr == '\0')) && (ctx->num_cpus < MAX_CPUS)) {
        if (cpu < 0) {
            ucs_error("invalid cpu number detected: (%d)", cpu);
            return UCS_ERR_INVALID_PARAM;
        }

        ctx->cpus[ctx->num_cpus++] = cpu;

        if (*endptr == '\0') {
            break;
        }

        cpu_list = endptr + 1; /* skip the comma */
        cpu      = strtol(cpu_list, &endptr, 10);
    }

    if (*endptr == ',') {
        ucs_error("number of listed cpus exceeds the maximum supported value (%d)",
                  MAX_CPUS);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t parse_opts(struct perftest_context *ctx, int mpi_initialized,
                               int argc, char **argv)
{
    ucs_status_t status;
    int c;

    ucs_trace_func("");

    ucx_perf_global_init(); /* initialize memory types */

    status = init_test_params(&ctx->params);
    if (status != UCS_OK) {
        return status;
    }

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
                ctx->batch_files[ctx->num_batch_files++] = optarg;
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
            status = parse_cpus(optarg, ctx);
            if (status != UCS_OK) {
                return status;
            }
            break;
        case 'P':
#ifdef HAVE_MPI
            ctx->mpi = atoi(optarg) && mpi_initialized;
            break;
#endif
        case 'h':
            usage(ctx, ucs_basename(argv[0]));
            return UCS_ERR_CANCELED;
        default:
            status = parse_test_params(&ctx->params, c, optarg);
            if (status != UCS_OK) {
                usage(ctx, ucs_basename(argv[0]));
                return status;
            }
            break;
        }
    }

    if (optind < argc) {
        ctx->server_addr = argv[optind];
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
#pragma omp barrier

#pragma omp master
  {
    sock_rte_group_t *group = rte_group;
    const unsigned magic = 0xdeadbeef;
    unsigned snc;

    snc = magic;
    safe_send(group->connfd, &snc, sizeof(unsigned), progress, arg);

    snc = 0;
    safe_recv(group->connfd, &snc, sizeof(unsigned), progress, arg);

    ucs_assert(snc == magic);
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
                            void *arg, int is_final, int is_multi_thread)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final, ctx->server_addr == NULL, is_multi_thread);
}

static ucx_perf_rte_t sock_rte = {
    .group_size    = sock_rte_group_size,
    .group_index   = sock_rte_group_index,
    .barrier       = sock_rte_barrier,
    .post_vec      = sock_rte_post_vec,
    .recv          = sock_rte_recv,
    .exchange_vec  = (ucx_perf_rte_exchange_vec_func_t)ucs_empty_function,
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
        status = ucs_socket_setopt(sockfd, SOL_SOCKET, SO_REUSEADDR,
                                   &optval, sizeof(optval));
        if (status != UCS_OK) {
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

        /* release the memory for the list of the message sizes allocated
         * during the initialization of the default testing parameters */
        free(ctx->params.super.msg_size_list);
        ctx->params.super.msg_size_list = NULL;

        ret = safe_recv(connfd, &ctx->params, sizeof(ctx->params), NULL, NULL);
        if (ret) {
            status = UCS_ERR_IO_ERROR;
            goto err_close_connfd;
        }

        if (ctx->params.super.msg_size_cnt != 0) {
            ctx->params.super.msg_size_list =
                    calloc(ctx->params.super.msg_size_cnt,
                           sizeof(*ctx->params.super.msg_size_list));
            if (NULL == ctx->params.super.msg_size_list) {
                status = UCS_ERR_NO_MEMORY;
                goto err_close_connfd;
            }

            ret = safe_recv(connfd, ctx->params.super.msg_size_list,
                            sizeof(*ctx->params.super.msg_size_list) *
                            ctx->params.super.msg_size_cnt,
                            NULL, NULL);
            if (ret) {
                status = UCS_ERR_IO_ERROR;
                goto err_close_connfd;
            }
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
        if (ctx->params.super.msg_size_cnt != 0) {
            safe_send(sockfd, ctx->params.super.msg_size_list,
                      sizeof(*ctx->params.super.msg_size_list) *
                      ctx->params.super.msg_size_cnt,
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

    ctx->params.super.rte_group  = &ctx->sock_rte_group;
    ctx->params.super.rte        = &sock_rte;
    ctx->params.super.report_arg = ctx;
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

#if defined (HAVE_MPI)
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

#pragma omp barrier

#pragma omp master
  {
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
                           void *arg, int is_final, int is_multi_thread)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final, ctx->server_addr == NULL, is_multi_thread);
}
#elif defined (HAVE_RTE)
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
#pragma omp barrier

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
                           void *arg, int is_final, int is_multi_thread)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final, ctx->server_addr == NULL, is_multi_thread);
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
#if defined (HAVE_MPI)
    static ucx_perf_rte_t mpi_rte = {
        .group_size    = mpi_rte_group_size,
        .group_index   = mpi_rte_group_index,
        .barrier       = mpi_rte_barrier,
        .post_vec      = mpi_rte_post_vec,
        .recv          = mpi_rte_recv,
        .exchange_vec  = (void*)ucs_empty_function,
        .report        = mpi_rte_report,
    };

    int size, rank;

    ucs_trace_func("");

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        ucs_error("This test should run with exactly 2 processes (actual: %d)", size);
        return UCS_ERR_INVALID_PARAM;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 1) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.super.rte_group  = NULL;
    ctx->params.super.rte        = &mpi_rte;
    ctx->params.super.report_arg = ctx;
#elif defined (HAVE_RTE)
    ucs_trace_func("");
    
    ctx->params.rte_group         = NULL;
    ctx->params.rte               = &mpi_rte;
    ctx->params.report_arg        = ctx;
    rte_group_t group;

    rte_init(NULL, NULL, &group);
    if (1 == rte_group_rank(group)) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.super.rte_group  = group;
    ctx->params.super.rte        = &ext_rte;
    ctx->params.super.report_arg = ctx;
#endif
    return UCS_OK;
}

static ucs_status_t cleanup_mpi_rte(struct perftest_context *ctx)
{
#ifdef HAVE_RTE
    rte_finalize();
#endif
    return UCS_OK;
}

static ucs_status_t check_system(struct perftest_context *ctx)
{
    ucs_sys_cpuset_t cpuset;
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
        for (i = 0; i < ctx->num_cpus; i++) {
            if (ctx->cpus[i] >= nr_cpus) {
                ucs_error("cpu (%u) out of range (0..%u)", ctx->cpus[i], nr_cpus - 1);
                return UCS_ERR_INVALID_PARAM;
            }
        }

        for (i = 0; i < ctx->num_cpus; i++) {
            CPU_SET(ctx->cpus[i], &cpuset);
        }

        ret = ucs_sys_setaffinity(&cpuset);
        if (ret) {
            ucs_warn("sched_setaffinity() failed: %m");
            return UCS_ERR_INVALID_PARAM;
        }
    } else {
        ret = ucs_sys_getaffinity(&cpuset);
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

static ucs_status_t clone_params(perftest_params_t *dest,
                                 const perftest_params_t *src)
{
    size_t msg_size_list_size;

    *dest                     = *src;
    msg_size_list_size        = dest->super.msg_size_cnt *
                                sizeof(*dest->super.msg_size_list);
    dest->super.msg_size_list = malloc(msg_size_list_size);
    if (dest->super.msg_size_list == NULL) {
        return ((msg_size_list_size != 0) ? UCS_ERR_NO_MEMORY : UCS_OK);
    }

    memcpy(dest->super.msg_size_list, src->super.msg_size_list,
           msg_size_list_size);
    return UCS_OK;
}

static ucs_status_t run_test_recurs(struct perftest_context *ctx,
                                    const perftest_params_t *parent_params,
                                    unsigned depth)
{
    perftest_params_t params;
    ucx_perf_result_t result;
    ucs_status_t status;
    FILE *batch_file;
    int line_num;

    ucs_trace_func("depth=%u, num_files=%u", depth, ctx->num_batch_files);

    if (parent_params->super.api == UCX_PERF_API_UCP) {
        if (strcmp(parent_params->super.uct.dev_name, TL_RESOURCE_NAME_NONE)) {
            ucs_warn("-d '%s' ignored for UCP test; see NOTES section in help message",
                     parent_params->super.uct.dev_name);
        }
        if (strcmp(parent_params->super.uct.tl_name, TL_RESOURCE_NAME_NONE)) {
            ucs_warn("-x '%s' ignored for UCP test; see NOTES section in help message",
                     parent_params->super.uct.tl_name);
        }
    }

    if (depth >= ctx->num_batch_files) {
        print_test_name(ctx);
        return ucx_perf_run(&parent_params->super, &result);
    }

    batch_file = fopen(ctx->batch_files[depth], "r");
    if (batch_file == NULL) {
        ucs_error("Failed to open batch file '%s': %m", ctx->batch_files[depth]);
        return UCS_ERR_IO_ERROR;
    }

    line_num = 0;
    do {
        status = clone_params(&params, parent_params);
        if (status != UCS_OK) {
            goto out;
        }

        status = read_batch_file(batch_file, ctx->batch_files[depth],
                                 &line_num, &params,
                                 &ctx->test_names[depth]);
        if (status == UCS_OK) {
            run_test_recurs(ctx, &params, depth + 1);
            free(ctx->test_names[depth]);
            ctx->test_names[depth] = NULL;
        }

        free(params.super.msg_size_list);
        params.super.msg_size_list = NULL;
    } while (status == UCS_OK);

    if (status == UCS_ERR_NO_ELEM) {
        status = UCS_OK;
    }

out:
    fclose(batch_file);
    return status;
}

static ucs_status_t run_test(struct perftest_context *ctx)
{
    const char *error_prefix;
    ucs_status_t status;

    ucs_trace_func("");

    setlocale(LC_ALL, "en_US");

    /* no batch files, only command line params */
    if (ctx->num_batch_files == 0) {
        error_prefix = (ctx->flags & TEST_FLAG_PRINT_RESULTS) ?
                       "command line: " : "";
        status       = adjust_test_params(&ctx->params, error_prefix);
        if (status != UCS_OK) {
            return status;
        }
    }

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

#ifdef HAVE_MPI
    int provided;

    mpi_initialized = !isatty(0) &&
                      /* Using MPI_THREAD_FUNNELED since ucx_perftest supports
                       * using multiple threads when only the main one makes
                       * MPI calls (which is also suitable for a single threaded
                       * run).
                       * MPI_THREAD_FUNNELED:
                       * The process may be multi-threaded, but only the main
                       * thread will make MPI calls (all MPI calls are funneled
                       * to the main thread). */
                      (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) == 0);

    if (mpi_initialized && (provided != MPI_THREAD_FUNNELED)) {
        printf("MPI_Init_thread failed to set MPI_THREAD_FUNNELED. (provided = %d)\n",
               provided);
        ret = -1;
        goto out;
    }
#else
    mpi_initialized = 0;
#endif

    /* Parse command line */
    status = parse_opts(&ctx, mpi_initialized, argc, argv);
    if (status != UCS_OK) {
        ret = (status == UCS_ERR_CANCELED) ? 0 : -127;
        goto out_msg_size_list;
    }

#ifdef __COVERITY__
    /* coverity[dont_call] */
    mpi_rte = rand(); /* Shut up deadcode error */
#endif

    if (ctx.mpi) {
        mpi_rte = 1;
    } else {
#ifdef HAVE_RTE
        mpi_rte = 1;
#else
        mpi_rte = 0;
#endif
    }

    status = check_system(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_msg_size_list;
    }

    /* Create RTE */
    status = (mpi_rte) ? setup_mpi_rte(&ctx) : setup_sock_rte(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_msg_size_list;
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
out_msg_size_list:
    free(ctx.params.super.msg_size_list);
#if HAVE_MPI
out:
#endif
    if (mpi_initialized) {
#ifdef HAVE_MPI
        MPI_Finalize();
#endif
    }
    return ret;
}
