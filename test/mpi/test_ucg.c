/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#define _GNU_SOURCE /* For basename */
#include <mpi.h>

#include <stdlib.h>
#include <unistd.h>

#include "ucg/base/ucg_group.h"
#include "tools/info/ucx_info.h"
#include "tools/info/group_info.c"

// TODO: add something like "gtest-mpi-listener" (LLNL's github)

/* Forward decleration for an OMPI's MCA component: coll/ucx */
ucg_worker_h mca_coll_ucx_get_component_worker();

enum {
    TEST_UCG_TOPOLOGY = UCS_BIT(0),
};

static void usage() {
    printf("Usage: test_ucg [options]\n");
    printf("Options are:\n");
    printf("  -c <type>   Select collective operation type by name.\n");
    printf("  -p <name>   Select planner component by name.\n");
    printf("  -r <rank#>  Select rank number to test (default: all).\n");
    printf("  -t          Print topology information.\n");
    printf("  -f          Print full topology information.\n");
    printf("  -h          Print this usage string.\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *collective_name = NULL;
    const char *planner_name = NULL;
    int my_rank, test_rank = -1;
    unsigned test_flags = 0;
    int is_full = 0;
    int ret = 0;
    int c;

    while ((c = getopt(argc, argv, "c:p:r:fth")) != -1) {
        switch (c) {
        case 'c':
            collective_name = optarg;
            break;
        case 'p':
            planner_name = optarg;
            break;
        case 'r':
            test_rank = atoi(optarg);
            break;
        case 'f':
            is_full = 1;
            break;
        case 't':
            test_flags |= TEST_UCG_TOPOLOGY;
            break;
        case 'h':
        default:
            usage();
            return -1;
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (test_flags & TEST_UCG_TOPOLOGY) {
        ucg_worker_h worker = mca_coll_ucx_get_component_worker();
        ucg_groups_t *ctx = UCG_WORKER_TO_GROUPS_CTX(worker);
        ucg_group_h group;
        ucs_list_for_each(group, &ctx->groups_head, list) {
            if ((test_rank == -1) || (my_rank == test_rank)) {
                const ucg_group_params_t *params = ucg_group_get_params((ucg_group_h)group);
                print_ucg_topology(planner_name, worker, 0, my_rank, collective_name,
                        params->distance, params->member_count, is_full);
            }
        }
    }

    MPI_Finalize();
    return ret;
}
