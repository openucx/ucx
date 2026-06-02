/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_MLX5_EXT_H_
#define UCT_IB_MLX5_EXT_H_

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include <stdint.h>

#include <ucs/type/status.h>
#include <ucs/sys/stubs.h>

#define UCT_IB_MLX5_EXT_NAME_MAX 32

/**
 * @brief Field mask bits for @ref uct_ib_mlx5_ext_qp_query_attr_t.
 * Selects which output fields to populate.
 */
enum uct_ib_mlx5_ext_qp_query_attr_field {
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN_LEN = UCS_BIT(0),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN_LEN = UCS_BIT(1),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN     = UCS_BIT(2),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN     = UCS_BIT(3),
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_QP_NUM       = UCS_BIT(4)
};

/**
 * @brief QP tx/rx token query output attributes.
 */
typedef struct uct_ib_mlx5_ext_qp_query_attr {
    uint64_t field_mask;   /**< Mask of valid fields in this structure, using bits from @ref uct_ib_mlx5_ext_qp_query_attr_field. */
    size_t   tx_token_len; /**< Length of the TX token in bytes. */
    size_t   rx_token_len; /**< Length of the RX token in bytes. */
    void     *tx_token;    /**< TX token pointer. */
    void     *rx_token;    /**< RX token pointer. */
    uint32_t qp_num;       /**< QP number. */
} uct_ib_mlx5_ext_qp_query_attr_t;


/**
 * @brief External plugin iface flags function type.
 */
typedef ucs_status_t (*uct_ib_mlx5_ext_iface_flags_func_t)(uint64_t *flags);

/**
 * @brief QP tx/rx token query function type.
 */
typedef ucs_status_t (*uct_ib_mlx5_ext_qp_query_func_t)(
        struct ibv_qp *qp, struct mlx5dv_devx_obj *devx_obj,
        uct_ib_mlx5_ext_qp_query_attr_t *attr);

typedef struct uct_ib_mlx5_ext_ops {
    char                               name[UCT_IB_MLX5_EXT_NAME_MAX];
    uct_ib_mlx5_ext_iface_flags_func_t iface_flags;
    uct_ib_mlx5_ext_qp_query_func_t    qp_query;
} uct_ib_mlx5_ext_ops_t;

/**
 * @brief Initialize mlx5 external extension.
 */
void uct_ib_mlx5_ext_init(void);

/**
 * @brief Release mlx5 external extension.
 */
void uct_ib_mlx5_ext_cleanup(void);

/**
 * @brief Register an external provider.
 *
 * @param [in] ops Pointer to the provider operations.
 */
void uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops);

/**
 * @brief Call the first registered provider supporting iface_flags.
 *
 * @param [out] flags Pointer to the iface flags.
 *
 * @return UCS_OK on success, error code otherwise.
 */
ucs_status_t uct_ib_mlx5_ext_iface_flags(uint64_t *flags);

/**
 * @brief Call the first registered provider supporting qp_query.
 *
 * @param [in]     qp          QP pointer.
 * @param [in]     devx_obj    DevX object pointer.
 * @param [in,out] attr        Output attributes.
 *
 * @return UCS_OK on success, error code otherwise.
 */
ucs_status_t uct_ib_mlx5_ext_qp_query(struct ibv_qp *qp,
                                      struct mlx5dv_devx_obj *devx_obj,
                                      uct_ib_mlx5_ext_qp_query_attr_t *attr);

#endif /* UCT_IB_MLX5_EXT_H_ */
