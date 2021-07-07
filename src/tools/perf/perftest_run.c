/**
* Copyright (C) NVIDIA 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "perftest.h"


void print_progress(char **test_names, unsigned num_names,
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
        fmt_numeric = "%'18.0f %10.3f %9.3f %9.3f %11.2f %10.2f %'11.0f %'11.0f\n";
        fmt_plain   = "%18.0f %10.3f %9.3f %9.3f %11.2f %10.2f %11.0f %11.0f\n";

        printf((flags & TEST_FLAG_PRINT_CSV)   ? fmt_csv :
               (flags & TEST_FLAG_NUMERIC_FMT) ? fmt_numeric :
                                                 fmt_plain,
               (double)result->iters,
               result->latency.percentile * 1000000.0,
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
            case UCT_PERF_DATA_LAYOUT_SHORT_IOV:
                test_data_str = "short iov";
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
        if ((test->api == UCX_PERF_API_UCP) &&
            (test->command == UCX_PERF_CMD_AM)) {
            printf("| AM header size: %-60zu             |\n",
                   ctx->params.super.ucp.am_hdr_size);
        }
    }

    if (ctx->flags & TEST_FLAG_PRINT_CSV) {
        if (ctx->flags & TEST_FLAG_PRINT_RESULTS) {
            for (i = 0; i < ctx->num_batch_files; ++i) {
                printf("%s,", ucs_basename(ctx->batch_files[i]));
            }
            printf("iterations,%.1f_percentile_lat,avg_lat,overall_lat,avg_bw,overall_bw,avg_mr,overall_mr\n", ctx->params.super.percentile_rank);
        }
    } else {
        if (ctx->flags & TEST_FLAG_PRINT_RESULTS) {
            overhead_lat_str = (test == NULL) ? "overhead" : test->overhead_lat;

            printf("+--------------+--------------+------------------------------+---------------------+-----------------------+\n");
            printf("|              |              |       %8s (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |\n", overhead_lat_str);
            printf("+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+\n");
            printf("|    Stage     | # iterations | %4.1f%%ile | average | overall |  average |  overall |  average  |  overall  |\n", ctx->params.super.percentile_rank);
            printf("+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+\n");
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
        strcpy(buf, "+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+");

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

void usage(const struct perftest_context *ctx, const char *program)
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
    printf("     -R <rank>      percentile rank of the percentile data in latency tests (%.1f)\n",
                                ctx->params.super.percentile_rank);
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
    printf("\n");
    printf("   NOTE: When running UCP tests, transport and device should be specified by\n");
    printf("         environment variables: UCX_TLS and UCX_[SELF|SHM|NET]_DEVICES.\n");
    printf("\n");
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

    if (depth >= ctx->num_batch_files) {
        print_test_name(ctx);
        status = check_params(parent_params);
        if (status != UCS_OK) {
            return status;
        }

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

ucs_status_t run_test(struct perftest_context *ctx)
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
