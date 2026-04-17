/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
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
 * Check that all SGL entries match the given memory info
 *
 * @param [in]     context        Context for memory detection
 * @param [in]     local          @ref ucp_dt_local_sgl_t descriptor to check
 * @param [in]     count          Number of entries in the @a local descriptor
 * @param [in]     mem_info       Compare the SGL entries to this memory info
 *
 * @return UCS_OK if all SGL entries match the given memory info, otherwise
 *         return UCS_ERR_INVALID_PARAM
 */
ucs_status_t ucp_dt_sgl_check_same_memtype(ucp_context_h context,
                                           const ucp_dt_local_sgl_t *local,
                                           size_t count,
                                           const ucp_memory_info_t *mem_info);

#endif
