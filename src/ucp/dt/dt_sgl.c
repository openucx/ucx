/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dt_sgl.h"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>


ucs_status_t ucp_dt_sgl_check_same_memtype(ucp_context_h context,
                                      const ucp_dt_local_sgl_t *local,
                                      size_t count,
                                      const ucp_memory_info_t *mem_info)
{
    ucs_status_t status;
    size_t i;

    for (i = 1; i < count; ++i) {
        status = ucp_dt_mem_type_check_elem(context, local->buffers[i],
                                            local->lengths[i], mem_info, "sgl",
                                            i, count);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}
