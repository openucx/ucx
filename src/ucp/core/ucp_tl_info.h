/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_TL_INFO_H_
#define UCP_TL_INFO_H_

#include <ucp/core/ucp_context.h>
#include <uct/api/uct.h>


typedef struct {
    uct_tl_resource_desc_t rsc;
    ucp_rsc_index_t        cmpt_index;
    int                    enabled;
} ucp_tl_info_entry_t;


void ucp_context_log_tl_info(ucp_context_h context,
                             const ucp_tl_info_entry_t *all_rscs,
                             unsigned num_all_rscs);


#endif
