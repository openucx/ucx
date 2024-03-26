/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021-2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_PERFTEST_H
#define UCX_PERFTEST_H

#include "api/libperf.h"
#include "lib/libperf_int.h"

#include "perftest_context.h"

#include <getopt.h>

#if defined (HAVE_MPI)
#  include <mpi.h>
#endif

#define TL_RESOURCE_NAME_NONE   "<none>"
#define TEST_PARAMS_ARGS        "t:n:s:W:O:w:D:i:H:oSCIqM:r:E:T:d:x:A:BUem:R:lyzg:G:"
#define TEST_ID_UNDEFINED       -1

#define DEFAULT_DAEMON_PORT     1338

typedef struct test_type {
    const char           *name;
    ucx_perf_api_t       api;
    ucx_perf_cmd_t       command;
    ucx_perf_test_type_t test_type;
    const char           *desc;
    const char           *overhead_lat;
    unsigned             window_size;
} test_type_t;

extern const struct option TEST_PARAMS_ARGS_LONG[];

extern test_type_t tests[];

ucs_status_t run_test(struct perftest_context *ctx);
ucs_status_t clone_params(perftest_params_t *dest,
                          const perftest_params_t *src);
ucs_status_t check_params(const perftest_params_t *params);
ucs_status_t parse_opts(struct perftest_context *ctx, int mpi_initialized,
                        int argc, char **argv);
ucs_status_t init_test_params(perftest_params_t *params);
ucs_status_t parse_test_params(perftest_params_t *params, char opt,
                               const char *opt_arg);
ucs_status_t adjust_test_params(perftest_params_t *params,
                                const char *error_prefix);
void print_progress(void *UCS_V_UNUSED rte_group,
                    const ucx_perf_result_t *result, void *arg,
                    const char *extra_info, int final, int is_multi_thread);

#endif /* UCX_PERFTEST_H */
