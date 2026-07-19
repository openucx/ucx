/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_TL_INFO_H_
#define UCP_TL_INFO_H_

#include <ucp/core/ucp_context.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/array.h>
#include <ucs/datastruct/string_buffer.h>


typedef struct {
    uct_tl_resource_desc_t rsc;
    ucp_rsc_index_t        cmpt_index;
    int                    enabled;
} ucp_tl_info_entry_t;


UCS_ARRAY_DECLARE_TYPE(ucp_tl_info_array_t, unsigned, ucp_tl_info_entry_t);


void ucp_context_log_tl_info(ucp_context_h context,
                             ucp_tl_info_array_t *all_rscs);


ucs_status_t ucp_context_render_tl_info(ucp_context_h context,
                                        ucp_tl_info_array_t *all_rscs,
                                        ucs_string_buffer_t *strb);


#endif
