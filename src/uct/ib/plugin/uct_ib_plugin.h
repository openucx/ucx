/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_PLUGIN_H_
#define UCT_IB_PLUGIN_H_

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>
#include <uct/api/v2/uct_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Plugin implementation of @ref uct_iface_query_v2.
 *
 * @param [in]     iface       Interface handle.
 * @param [in,out] iface_attr  Iface v2 attributes.
 *
 * @return UCS_OK on success, error code otherwise.
 */
ucs_status_t
uct_ib_plugin_iface_query(uct_iface_h iface, uct_iface_attr_v2_t *iface_attr);


/**
 * @brief Plugin implementation of @ref uct_ep_query.
 *
 * @param [in]     ep       Endpoint handle.
 * @param [in,out] ep_attr  Endpoint attributes.
 *
 * @return UCS_OK on success, error code otherwise.
 */
ucs_status_t uct_ib_plugin_ep_query(uct_ep_h ep, uct_ep_attr_t *ep_attr);


#ifdef __cplusplus
}
#endif

#endif /* UCT_IB_PLUGIN_H_ */
