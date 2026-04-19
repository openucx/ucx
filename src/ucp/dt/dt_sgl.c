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
#include <ucp/core/ucp_rkey.h>


ucs_status_t ucp_dt_sgl_check_same_mem_info(ucp_context_h context,
                                            const ucp_dt_local_sgl_t *local,
                                            size_t count,
                                            const ucp_memory_info_t *mem_info)
{
    ucs_status_t status;
    size_t i;

    for (i = 1; i < count; ++i) {
        status = ucp_dt_mem_info_check_elem(context, local->buffers[i],
                                            local->lengths[i], mem_info, "sgl",
                                            i, count);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t ucp_dt_sgl_check_same_rkey_config(const ucp_rkey_h *rkeys,
                                               size_t count)
{
    ucp_rkey_h ref_rkey = rkeys[0];
    ucp_rkey_h rkey;
    size_t i;

    if (ucs_unlikely(ref_rkey == NULL)) {
        ucs_error("sgl[0]: rkey is NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 1; i < count; ++i) {
        rkey = rkeys[i];
        if (ucs_unlikely(rkey == NULL)) {
            ucs_error("sgl[%zu]: rkey is NULL", i);
            return UCS_ERR_INVALID_PARAM;
        }

#if ENABLE_PARAMS_CHECK
        if (ucs_unlikely(rkey->ep != ref_rkey->ep)) {
            ucs_error("sgl[%zu]: rkey %p was unpacked on ep %p, but sgl[0] "
                      "rkey %p was unpacked on ep %p, all rkeys must belong "
                      "to the same endpoint",
                      i, rkey, rkey->ep, ref_rkey, ref_rkey->ep);
            return UCS_ERR_INVALID_PARAM;
        }
#endif

        if (ucs_unlikely(rkey->cfg_index != ref_rkey->cfg_index)) {
            ucs_error("sgl[%zu]: rkey %p has cfg_index %u, but sgl[0] rkey %p "
                      "has cfg_index %u, all rkeys must map to the same "
                      "remote endpoint configuration",
                      i, rkey, rkey->cfg_index, ref_rkey,
                      ref_rkey->cfg_index);
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}
