/**
* Copyright (C) NVIDIA 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "perftest.h"

#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/sock.h>
#include <ucs/debug/log.h>


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
    printf("     -O <count>     maximal number of uncompleted outstanding sends (%u)\n",
                                ctx->params.super.max_outstanding);
    printf("     -i <offset>    distance between consecutive scatter-gather entries (%zu)\n",
                                ctx->params.super.iov_stride);
    printf("     -l             use loopback connection\n");
    printf("                    in this case, the process will communicate with itself,\n");
    printf("                    so passing server hostname is not allowed\n");
    printf("     -o             do not progress the responder in one-sided tests\n");
    printf("     -B             register memory with NONBLOCK flag\n");
    printf("     -b <file>      read and execute tests from a batch file: every line in the\n");
    printf("                    file is a test to run, first word is test name, the rest of\n");
    printf("                    the line is command-line arguments for the test.\n");
    printf("     -R <rank>      percentile rank of the percentile data in latency tests (%.1f)\n",
                                ctx->params.super.percentile_rank);
    printf("     -p <port>      TCP port to use for data exchange (%d)\n", ctx->port);
    printf("     -6             Use IPv6 address for in data exchange\n");
#ifdef HAVE_MPI
    printf("     -P <0|1>       disable/enable MPI mode (%d)\n", ctx->mpi);
#endif
    printf("     -h             show this help message\n");
    printf("\n");
    printf("  Output format:\n");
    printf("     -N             use numeric formatting (thousands separator)\n");
    printf("     -f             print only final numbers\n");
    printf("     -v             print CSV-formatted output\n");
    printf("     -I             print extra information about the operation\n");
    printf("\n");
    printf("  UCT only:\n");
    printf("     -d <device>    device to use for testing\n");
    printf("     -x <tl>        transport to use for testing\n");
    printf("     -D <layout>    data layout for sender side:\n");
    printf("                        short    - short messages (default, cannot be used for get)\n");
    printf("                        shortiov - short io-vector messages (only for active messages)\n");
    printf("                        bcopy    - copy-out (cannot be used for atomics)\n");
    printf("                        zcopy    - zero-copy (cannot be used for atomics)\n");
    printf("     -W <count>     flow control window size, for active messages (%u)\n",
                                ctx->params.super.uct.fc_window);
    printf("     -H <size>      active message header size (%zu), included in message size\n",
                                ctx->params.super.uct.am_hdr_size);
    printf("     -A <mode>      asynchronous progress mode (thread_spinlock)\n");
    printf("                        thread_spinlock - separate progress thread with spin locking\n");
    printf("                        thread_mutex - separate progress thread with mutex locking\n");
    printf("                        signal - signal-based timer\n");
    printf("\n");
    printf("  UCP only:\n");
    printf("     -T <threads>   number of threads in the test (%d)\n",
           ctx->params.super.thread_count);
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
    printf("     -e             create endpoints with error handling support\n");
    printf("     -E <mode>      wait mode for tests\n");
    printf("                        poll       : repeatedly call worker_progress\n");
    printf("                        sleep      : go to sleep after posting requests\n");
    printf("     -H <size>      active message header size (%zu), not included in message size\n",
                                ctx->params.super.ucp.am_hdr_size);
    printf("     -z             pass pre-registered memory handle\n");
    printf("\n");
    printf("   NOTE: When running UCP tests, transport and device should be specified by\n");
    printf("         environment variables: UCX_TLS and UCX_[SELF|SHM|NET]_DEVICES.\n");
    printf("\n");
}

static ucs_status_t parse_mem_type(const char *opt_arg,
                                   ucs_memory_type_t *mem_type)
{
    ucs_memory_type_t it;

    if (opt_arg == NULL) {
        ucs_error("memory type string is NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    for (it = UCS_MEMORY_TYPE_HOST; it < UCS_MEMORY_TYPE_LAST; it++) {
        if(!strcmp(opt_arg, ucs_memory_type_names[it]) &&
           (ucx_perf_mem_type_allocators[it] != NULL)) {
            *mem_type = it;
            return UCS_OK;
        }
    }

    ucs_error("unsupported memory type: \"%s\"", opt_arg);
    return UCS_ERR_INVALID_PARAM;
}

static ucs_status_t parse_mem_type_params(const char *opt_arg,
                                          ucs_memory_type_t *send_mem_type,
                                          ucs_memory_type_t *recv_mem_type)
{
    const char *delim   = ",";
    char *token         = strtok((char*)opt_arg, delim);
    ucs_status_t status;

    status = parse_mem_type(token, send_mem_type);
    if (status != UCS_OK) {
        return status;
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

ucs_status_t parse_test_params(perftest_params_t *params, char opt,
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
        } else if (!strcmp(opt_arg, "shortiov")) {
            params->super.uct.data_layout   = UCT_PERF_DATA_LAYOUT_SHORT_IOV;
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
    case 'E':
        if (!strcmp(opt_arg, "poll")) {
            params->super.wait_mode = UCX_PERF_WAIT_MODE_POLL;
            return UCS_OK;
        } else if (!strcmp(opt_arg, "sleep")) {
            params->super.wait_mode = UCX_PERF_WAIT_MODE_SLEEP;
            return UCS_OK;
        } else {
            ucs_error("Invalid option argument for -E");
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;
    case 'i':
        params->super.iov_stride = atol(opt_arg);
        return UCS_OK;
    case 'l':
        params->super.flags |= UCX_PERF_TEST_FLAG_LOOPBACK;
        return UCS_OK;
    case 'n':
        params->super.max_iter = atol(opt_arg);
        return UCS_OK;
    case 's':
        return parse_message_sizes_params(opt_arg, &params->super);
    case 'H':
        params->super.uct.am_hdr_size = atol(opt_arg);
        params->super.ucp.am_hdr_size = atol(opt_arg);
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
    case 'e':
        params->super.flags |= UCX_PERF_TEST_FLAG_ERR_HANDLING;
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
    case 'R':
        params->super.percentile_rank = atof(opt_arg);
        if ((0.0 <= params->super.percentile_rank) && (params->super.percentile_rank <= 100.0)) {
            return UCS_OK;
        } else {
            ucs_error("Invalid option argument for -R");
            return UCS_ERR_INVALID_PARAM;
        }
    case 'm':
        if (UCS_OK != parse_mem_type_params(opt_arg,
                                            &params->super.send_mem_type,
                                            &params->super.recv_mem_type)) {
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;
    case 'z':
        params->super.flags |= UCX_PERF_TEST_FLAG_PREREG;
        return UCS_OK;
    default:
       return UCS_ERR_INVALID_PARAM;
    }
}

ucs_status_t adjust_test_params(perftest_params_t *params,
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

ucs_status_t clone_params(perftest_params_t *dest,
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

ucs_status_t check_params(const perftest_params_t *params)
{
    switch (params->super.api) {
    case UCX_PERF_API_UCT:
        if (!strcmp(params->super.uct.dev_name, TL_RESOURCE_NAME_NONE)) {
            ucs_error("A device must be specified with -d flag for UCT test");
            return UCS_ERR_INVALID_PARAM;
        }
        if (!strcmp(params->super.uct.tl_name, TL_RESOURCE_NAME_NONE)) {
            ucs_error(
                    "A transport must be specified with -x flag for UCT test");
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;
    case UCX_PERF_API_UCP:
        if (strcmp(params->super.uct.dev_name, TL_RESOURCE_NAME_NONE)) {
            ucs_warn("-d '%s' ignored for UCP test; see NOTES section in help "
                     "message",
                     params->super.uct.dev_name);
        }
        if (strcmp(params->super.uct.tl_name, TL_RESOURCE_NAME_NONE)) {
            ucs_warn("-x '%s' ignored for UCP test; see NOTES section in help "
                     "message",
                     params->super.uct.tl_name);
        }
        return UCS_OK;
    default:
        ucs_error("Invalid test case");
        return UCS_ERR_INVALID_PARAM;
    }
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

ucs_status_t parse_opts(struct perftest_context *ctx, int mpi_initialized,
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

    ctx->server_addr     = NULL;
    ctx->num_batch_files = 0;
    ctx->port            = 13337;
    ctx->af              = AF_INET;
    ctx->flags           = 0;
    ctx->mpi             = mpi_initialized;

    optind = 1;
    while ((c = getopt(argc, argv, "p:b:6NfvIc:P:h" TEST_PARAMS_ARGS)) != -1) {
        switch (c) {
        case 'p':
            ctx->port = atoi(optarg);
            break;
        case '6':
            ctx->af = AF_INET6;
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
        case 'I':
            ctx->flags |= TEST_FLAG_PRINT_EXTRA_INFO;
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

        if (ctx->params.super.flags & UCX_PERF_TEST_FLAG_LOOPBACK) {
            ucs_error("conflicting arguments: server hostname argument is not "
                      "allowed in loopback (-l) mode");
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}
