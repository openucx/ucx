/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_SGL_H_
#define UCP_DT_SGL_H_

#include <ucp/api/ucp.h>
#include <ucp/dt/dt.h>


#define UCP_DT_IS_SGL(_datatype) \
    (((_datatype) & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_SGL)


/**
 * Check that all SGL entries match the given memory info.
 *
 * When memhs are provided, compares memhs[i]->mem_type and sys_dev.
 * When memhs are NULL, calls ucp_memory_detect on each buffer.
 *
 * @param [in]     context        Context for memory detection
 * @param [in]     local          Local SGL descriptor
 * @param [in]     count          Number of SGL entries
 * @param [in]     mem_info       Compare the SGL entries to this memory info
 *
 * @return UCS_OK if all entries match, otherwise UCS_ERR_INVALID_PARAM
 */
ucs_status_t ucp_dt_sgl_memtype_check(ucp_context_h context,
                                      const ucp_dt_local_sgl_t *local,
                                      size_t count,
                                      const ucp_memory_info_t *mem_info);

#endif
