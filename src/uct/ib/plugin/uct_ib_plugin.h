/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_PLUGIN_H_
#define UCT_IB_PLUGIN_H_

#include <stdint.h>
#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>
#include <uct/api/v2/uct_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    UCT_IB_PLUGIN_QP_VERBS,
    UCT_IB_PLUGIN_QP_DEVX
} uct_ib_plugin_qp_type_t;

typedef struct {
    uct_ib_plugin_qp_type_t type;
    uint32_t                qp_num;
    struct ibv_qp           *verbs_qp;
    struct mlx5dv_devx_obj  *devx_obj;
} uct_ib_plugin_qp_ctx_t;

/**
 * Return UCT iface capability flags contributed by the plugin.
 *
 * @return Bitmask of UCT_IFACE_FLAG_* to OR into iface_attr->cap.flags.
 */
uint64_t uct_ib_plugin_iface_flags(void);

/**
 * Opaque completion token query function.
 *
 * @param [in]     qp_ctx   Light-weight QP context.
 * @param [in,out] ep_attr  Endpoint attributes; field_mask selects which
 *                          token fields to fill (tx_token / rx_token).
 *
 * @return UCS_OK on success, error code otherwise.
 */
ucs_status_t uct_ib_plugin_query_token(const uct_ib_plugin_qp_ctx_t *qp_ctx,
                                       uct_ep_attr_t *ep_attr);


#ifdef __cplusplus
}
#endif

#endif /* UCT_IB_PLUGIN_H_ */
