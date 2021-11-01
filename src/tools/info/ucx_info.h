/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_INFO_H
#define UCX_INFO_H

#include <ucs/sys/sock.h>
#include <uct/api/uct.h>
#include <ucp/api/ucp.h>

#include <arpa/inet.h>


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
    PRINT_SYS_TOPO       = UCS_BIT(9),
    PRINT_MEMCPY_BW      = UCS_BIT(10)
};


typedef enum {
    PROCESS_PLACEMENT_SELF,
    PROCESS_PLACEMENT_INTRA,
    PROCESS_PLACEMENT_INTER
} process_placement_t;


void print_version();

void print_sys_info(int print_opts);

void print_build_config();

void print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                    const char *req_tl_name);

void print_type_info(const char * tl_name);

ucs_status_t
print_ucp_info(int print_opts, ucs_config_print_flags_t print_flags,
               uint64_t ctx_features, const ucp_ep_params_t *base_ep_params,
               size_t estimated_num_eps, size_t estimated_num_ppn,
               unsigned dev_type_bitmap, process_placement_t proc_placement,
               const char *mem_size, const char *ip_addr, sa_family_t af);

#endif
