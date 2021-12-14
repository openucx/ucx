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

#include <getopt.h>
#include <string.h>
#include <locale.h>


void print_progress(char **test_names, unsigned num_names,
                    const ucx_perf_result_t *result, const char *extra_info,
                    unsigned flags, int final, int is_server,
                    int is_multi_thread)
{
    UCS_STRING_BUFFER_ONSTACK(test_name, 128);
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

    if (!final) {
#if _OPENMP
        printf("[thread %d]", omp_get_thread_num());
#endif
    } else if ((flags & TEST_FLAG_PRINT_RESULTS) &&
               !(flags & TEST_FLAG_PRINT_CSV)) {
        if (flags & TEST_FLAG_PRINT_FINAL) {
            /* Print test name in the final and only output line */
            for (i = 0; i < num_names; ++i) {
                ucs_string_buffer_appendf(&test_name, "%s/", test_names[i]);
            }
            ucs_string_buffer_rtrim(&test_name, "/");
            printf("%10s", ucs_string_buffer_cstr(&test_name));
        } else {
            printf("Final:    ");
        }
    }

    if (is_multi_thread && final) {
        fmt_csv     = "%4.0f,%.3f,%.2f,%.0f";
        fmt_numeric = "%'18.0f %29.3f %22.2f %'24.0f";
        fmt_plain   = "%18.0f %29.3f %22.2f %23.0f";

        printf((flags & TEST_FLAG_PRINT_CSV)   ? fmt_csv :
               (flags & TEST_FLAG_NUMERIC_FMT) ? fmt_numeric :
                                                 fmt_plain,
               (double)result->iters,
               result->latency.total_average * 1000000.0,
               result->bandwidth.total_average / (1024.0 * 1024.0),
               result->msgrate.total_average);
    } else {
        fmt_csv     = "%4.0f,%.3f,%.3f,%.3f,%.2f,%.2f,%.0f,%.0f";
        fmt_numeric = "%'18.0f %10.3f %9.3f %9.3f %11.2f %10.2f %'11.0f %'11.0f";
        fmt_plain   = "%18.0f %10.3f %9.3f %9.3f %11.2f %10.2f %11.0f %11.0f";

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

    if ((flags & TEST_FLAG_PRINT_EXTRA_INFO) &&
        !(flags & TEST_FLAG_PRINT_CSV)) {
        printf("  %s", extra_info);
    }
    printf("\n");
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

        printf("+----------------------------------------------------------------------------------------------------------+\n");
        printf("| API:          %-60s                               |\n", test_api_str);
        printf("| Test:         %-60s                               |\n", test->desc);
        printf("| Data layout:  %-60s                               |\n", test_data_str);
        printf("| Send memory:  %-60s                               |\n", ucs_memory_type_names[ctx->params.super.send_mem_type]);
        printf("| Recv memory:  %-60s                               |\n", ucs_memory_type_names[ctx->params.super.recv_mem_type]);
        printf("| Message size: %-60zu                               |\n", ucx_perf_get_message_size(&ctx->params.super));
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
            printf("|    %5s     | # iterations | %4.1f%%ile | average | overall |  average |  overall |  average  |  overall  |\n",
                   (ctx->flags & TEST_FLAG_PRINT_FINAL) ? "Test" : "Stage",
                   ctx->params.super.percentile_rank);
            printf("+--------------+--------------+----------+---------+---------+----------+----------+-----------+-----------+\n");
        } else if (ctx->flags & TEST_FLAG_PRINT_TEST) {
            printf("+----------------------------------------------------------------------------------------------------------+\n");
        }
    }
}

static void print_test_name(struct perftest_context *ctx)
{
    char buf[200];
    unsigned i, pos;

    if (!(ctx->flags & (TEST_FLAG_PRINT_CSV | TEST_FLAG_PRINT_FINAL)) &&
        (ctx->num_batch_files > 0)) {
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
