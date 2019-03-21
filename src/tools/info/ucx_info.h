/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_INFO_H
#define UCX_INFO_H

#include <uct/api/uct.h>
#include <ucp/api/ucp.h>
#include <strings.h>

enum {
    PRINT_VERSION        = UCS_BIT(0),
    PRINT_SYS_INFO       = UCS_BIT(1),
    PRINT_BUILD_CONFIG   = UCS_BIT(2),
    PRINT_TYPES          = UCS_BIT(3),
    PRINT_DEVICES        = UCS_BIT(4),
    PRINT_UCP_CONTEXT    = UCS_BIT(5),
    PRINT_UCP_WORKER     = UCS_BIT(6),
    PRINT_UCP_EP         = UCS_BIT(7)

};


void print_version(void);

void print_sys_info(void);

void print_build_config(void);

void print_uct_info(int print_opts, ucs_config_print_flags_t print_flags,
                    const char *req_tl_name);

void print_type_info(const char * tl_name);

void print_ucp_info(int print_opts, ucs_config_print_flags_t print_flags,
                    uint64_t ctx_features, const ucp_ep_params_t *base_ep_params,
                    size_t estimated_num_eps, unsigned dev_type_bitmap);

#endif
