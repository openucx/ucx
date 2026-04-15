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


ucs_status_t ucp_dt_sgl_memtype_check(ucp_context_h context,
                                      const ucp_dt_local_sgl_t *local,
                                      size_t count,
                                      const ucp_memory_info_t *mem_info)
{
    ucp_memory_info_t mem_info_i;
    size_t i;

    for (i = 1; i < count; ++i) {
        ucp_memory_detect(context, local->buffers[i], local->lengths[i],
                          &mem_info_i);
        if ((mem_info_i.type != mem_info->type) ||
            (mem_info_i.sys_dev != mem_info->sys_dev)) {
            ucs_error("inconsistent sgl memtypes: "
                      "buffers[%zu]=%s-%s buffers[0]=%s-%s count=%zu",
                      i,
                      ucs_memory_type_names[mem_info_i.type],
                      ucs_topo_sys_device_get_name(mem_info_i.sys_dev),
                      ucs_memory_type_names[mem_info->type],
                      ucs_topo_sys_device_get_name(mem_info->sys_dev),
                      count);
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}
