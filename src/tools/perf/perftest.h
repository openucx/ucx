/**
* Copyright (C) NVIDIA 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_PERFTEST_H
#define UCX_PERFTEST_H

#include "api/libperf.h"
#include "lib/libperf_int.h"

#if defined (HAVE_MPI)
#  include <mpi.h>
#elif defined (HAVE_RTE)
#   include<rte.h>
#endif


#define MAX_BATCH_FILES         32
#define MAX_CPUS                1024
#define TL_RESOURCE_NAME_NONE   "<none>"
#define TEST_PARAMS_ARGS        "t:n:s:W:O:w:D:i:H:oSCIqM:r:E:T:d:x:A:BUem:R:lz"
#define TEST_ID_UNDEFINED       -1

enum {
    TEST_FLAG_PRINT_RESULTS    = UCS_BIT(0),
    TEST_FLAG_PRINT_TEST       = UCS_BIT(1),
    TEST_FLAG_SET_AFFINITY     = UCS_BIT(8),
    TEST_FLAG_NUMERIC_FMT      = UCS_BIT(9),
    TEST_FLAG_PRINT_FINAL      = UCS_BIT(10),
    TEST_FLAG_PRINT_CSV        = UCS_BIT(11),
    TEST_FLAG_PRINT_EXTRA_INFO = UCS_BIT(12)
};

typedef struct sock_rte_group {
    int                          sendfd;
    int                          recvfd;
    int                          is_server;
    int                          size;
    int                          peer;
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
    sa_family_t                  af;
    int                          mpi;
    unsigned                     num_cpus;
    unsigned                     cpus[MAX_CPUS];
    unsigned                     flags;

    unsigned                     num_batch_files;
    char                         *batch_files[MAX_BATCH_FILES];
    char                         *test_names[MAX_BATCH_FILES];

    sock_rte_group_t             sock_rte_group;
};

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
void print_progress(char **test_names, unsigned num_names,
                    const ucx_perf_result_t *result, const char *extra_info,
                    unsigned flags, int final, int is_server,
                    int is_multi_thread);

#endif /* UCX_PERFTEST_H */
