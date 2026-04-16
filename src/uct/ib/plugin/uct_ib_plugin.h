/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_PLUGIN_H_
#define UCT_IB_PLUGIN_H_

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>

#include <stdint.h>
#include <stddef.h>

struct ibv_context;
struct ibv_qp;
struct mlx5dv_devx_obj;

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Field mask bits for @ref uct_ib_plugin_qp_query_params_t.
 * Indicates which input fields are valid.
 */
enum uct_ib_plugin_qp_query_param_field {
    UCT_IB_PLUGIN_QP_QUERY_PARAM_FIELD_CTX      = UCS_BIT(0),
    UCT_IB_PLUGIN_QP_QUERY_PARAM_FIELD_QP       = UCS_BIT(1),
    UCT_IB_PLUGIN_QP_QUERY_PARAM_FIELD_DEVX_OBJ = UCS_BIT(2),
    UCT_IB_PLUGIN_QP_QUERY_PARAM_FIELD_QP_NUM   = UCS_BIT(3)
};

/**
 * @brief QP query input parameters.
 */
typedef struct uct_ib_plugin_qp_query_params {
    uint64_t               field_mask;
    struct ibv_context     *ctx;
    struct ibv_qp          *qp;
    struct mlx5dv_devx_obj *devx_obj;
    uint32_t               qp_num;
} uct_ib_plugin_qp_query_params_t;

/**
 * @brief Field mask bits for @ref uct_ib_plugin_qp_query_attr_t.
 * Selects which output fields to populate.
 */
enum uct_ib_plugin_qp_query_attr_field {
    UCT_IB_PLUGIN_QP_QUERY_ATTR_FIELD_TX_TOKEN_LEN = UCS_BIT(0),
    UCT_IB_PLUGIN_QP_QUERY_ATTR_FIELD_RX_TOKEN_LEN = UCS_BIT(1),
    UCT_IB_PLUGIN_QP_QUERY_ATTR_FIELD_TX_TOKEN     = UCS_BIT(2),
    UCT_IB_PLUGIN_QP_QUERY_ATTR_FIELD_RX_TOKEN     = UCS_BIT(3)
};

/**
 * @brief QP query output attributes.
 */
typedef struct uct_ib_plugin_qp_query_attr {
    uint64_t field_mask;
    size_t   tx_token_len;
    size_t   rx_token_len;
    void     *tx_token;
    void     *rx_token;
} uct_ib_plugin_qp_query_attr_t;


/**
 * @brief Return plugin-contributed iface capability flags.
 *
 * @return Bitmask of UCT_IFACE_FLAG_* contributed by the plugin.
 */
uint64_t uct_ib_plugin_iface_flags(void);

/**
 * @brief Query QP token information.
 *
 * @param [in]     params  Input parameters.  @a params->field_mask selects
 *                         which fields (ctx, qp, devx_obj) are valid.
 * @param [in,out] attr    Output attributes.  @a attr->field_mask selects
 *                         which fields to populate.  When
 *                         UCT_IB_PLUGIN_QP_QUERY_ATTR_FIELD_TX_TOKEN /
 *                         UCT_IB_PLUGIN_QP_QUERY_ATTR_FIELD_RX_TOKEN is set,
 *                         attr->tx_token / attr->rx_token must point to a
 *                         caller-allocated buffer of the appropriate length.
 *
 * @return UCS_OK on success, error code otherwise.
 */
ucs_status_t
uct_ib_plugin_qp_query(const uct_ib_plugin_qp_query_params_t *params,
                       uct_ib_plugin_qp_query_attr_t *attr);


#ifdef __cplusplus
}
#endif

#endif /* UCT_IB_PLUGIN_H_ */
