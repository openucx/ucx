/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_PERFTEST_CONTEXT_H
#define UCX_PERFTEST_CONTEXT_H

#include "api/libperf.h"
#include "lib/libperf_int.h"
#include <ucs/sys/string.h>


#define MAX_BATCH_FILES         32
#define MAX_CPUS                1024


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


typedef struct perftest_params {
    ucx_perf_params_t            super;
    int                          test_id;
} perftest_params_t;


struct perftest_context {
    perftest_params_t            params;
    const char                   *server_addr;
    uint16_t                     port;
    sa_family_t                  af;
    int                          mpi;
    unsigned                     num_cpus;
    unsigned                     cpus[MAX_CPUS];
    unsigned                     flags;

    unsigned                     num_batch_files;
    char                         *batch_files[MAX_BATCH_FILES];
    char                         *test_names[MAX_BATCH_FILES];
    const char                   *mad_port;

    sock_rte_group_t             sock_rte_group;
};


static inline void
perftest_params_release_msg_size_list(perftest_params_t *params)
{
    free(params->super.msg_size_list);
    params->super.msg_size_list = NULL;
}


static inline ucs_status_t
perftest_params_merge(perftest_params_t *dst, const perftest_params_t *src)
{
    char uct_dev_name[UCT_DEVICE_NAME_MAX];
    unsigned needed_flags;

    /* backup required dst parameters */
    ucs_strncpy_safe(uct_dev_name, dst->super.uct.dev_name,
                     UCT_DEVICE_NAME_MAX);
    needed_flags = (dst->super.flags & UCX_PERF_TEST_FLAG_ERR_HANDLING);

    perftest_params_release_msg_size_list(dst);

    memcpy(dst, src, sizeof(*src));

    if (dst->super.msg_size_cnt != 0) {
        dst->super.msg_size_list = calloc(dst->super.msg_size_cnt,
                                          sizeof(*dst->super.msg_size_list));
        if (dst->super.msg_size_list == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        memcpy(dst->super.msg_size_list, src->super.msg_size_list,
               src->super.msg_size_cnt * sizeof(*src->super.msg_size_list));
    }

    /* restore required overwritten parameters */
    ucs_strncpy_safe(dst->super.uct.dev_name, uct_dev_name,
                     UCT_DEVICE_NAME_MAX);
    dst->super.flags |= needed_flags;
    return UCS_OK;
}


extern ucs_list_link_t rte_list;

#endif /* UCX_PERFTEST_CONTEXT_H */
