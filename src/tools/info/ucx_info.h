/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
/**
*2019.12.30-Changed process for coll_ucx
*        Huawei Technologies Co., Ltd. 2019.
*/


#ifndef UCX_INFO_H
#define UCX_INFO_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/api/uct.h>
#include <ucp/api/ucp.h>
#if ENABLE_UCG
#include <ucg/api/ucg.h>
#endif


enum {
    PRINT_VERSION        = UCS_BIT(0),
    PRINT_SYS_INFO       = UCS_BIT(1),
    PRINT_BUILD_CONFIG   = UCS_BIT(2),
    PRINT_TYPES          = UCS_BIT(3),
    PRINT_DEVICES        = UCS_BIT(4),
    PRINT_UCP_CONTEXT    = UCS_BIT(5),
    PRINT_UCP_WORKER     = UCS_BIT(6),
    PRINT_UCP_EP         = UCS_BIT(7),
    PRINT_MEM_MAP        = UCS_BIT(8),
    PRINT_UCG            = UCS_BIT(9),
    PRINT_UCG_TOPO       = UCS_BIT(10)
};


void print_version();

void print_sys_info();

void print_build_config();

void print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                    const char *req_tl_name);

void print_type_info(const char * tl_name);

void print_ucp_info(int print_opts, ucs_config_print_flags_t print_flags,
                    uint64_t ctx_features, const ucp_ep_params_t *base_ep_params,
                    size_t estimated_num_eps, size_t estimated_num_ppn,
                    unsigned dev_type_bitmap, const char *mem_size
#if ENABLE_UCG
                    ,const char *planner_name,
                    ucg_group_member_index_t root_index,
                    ucg_group_member_index_t my_index,
                    const char *collective_type_name,
                    ucg_group_member_index_t peer_count[UCG_GROUP_MEMBER_DISTANCE_LAST]);

ucs_status_t gen_ucg_topology(ucg_group_member_index_t me,
        ucg_group_member_index_t peer_count[UCG_GROUP_MEMBER_DISTANCE_LAST],
        enum ucg_group_member_distance **distance_array_p,
        ucg_group_member_index_t *distance_array_length_p);

void print_ucg_topology(const char *req_planner_name, ucg_worker_h worker,
        ucg_group_member_index_t root,
        ucg_group_member_index_t me,
        const char *collective_type_name,
        enum ucg_group_member_distance *distance_array,
        ucg_group_member_index_t member_count, int is_verbose);
#else
                    );
#endif /* ENABLE_UCG */

#endif
