/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_MLX5_EXT_H_
#define UCT_IB_MLX5_EXT_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#if HAVE_MLX5_DV
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#else
struct ibv_qp;
struct mlx5dv_devx_obj;
#endif

#include <stdint.h>

#include <uct/api/uct_def.h>
#include <ucs/type/status.h>
#include <ucs/sys/stubs.h>

BEGIN_C_DECLS

/**
 * @brief QP token query attributes field mask.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_ib_mlx5_ext_qp_query_attr_t are present.
 */
enum uct_ib_mlx5_ext_qp_query_attr_field {
    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::tx_token_len */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN_LEN = UCS_BIT(0),

    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::rx_token_len */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN_LEN = UCS_BIT(1),

    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::tx_token */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN     = UCS_BIT(2),

    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::rx_token */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN     = UCS_BIT(3),

    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::qp_num */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_QP_NUM       = UCS_BIT(4)
};

/**
 * @brief QP token query parameters.
 */
typedef struct uct_ib_mlx5_ext_qp_query_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_ib_mlx5_ext_qp_query_attr_field. Fields not specified in this
     * mask will be ignored.
     */
    uint64_t field_mask;

    /** TX token length in bytes. */
    size_t   tx_token_len;

    /** RX token length in bytes. */
    size_t   rx_token_len;

    /**
     * Pointer to a caller-allocated buffer for TX token data. The buffer size
     * must be at least @ref tx_token_len bytes.
     */
    void     *tx_token;

    /**
     * Pointer to a caller-allocated buffer for RX token data. The buffer size
     * must be at least @ref rx_token_len bytes.
     */
    void     *rx_token;

    /**
     * QP number. Required as input when querying via a DevX object and
     * @a qp is NULL.
     */
    uint32_t qp_num;
} uct_ib_mlx5_ext_qp_query_attr_t;


/**
 * @brief External provider iface flags callback.
 *
 * @param [out] flags Interface capability flags to fill.
 *
 * @return UCS_OK on success, or an error if the operation failed.
 */
typedef ucs_status_t (*uct_ib_mlx5_ext_iface_flags_func_t)(uint64_t *flags);

/**
 * @brief External provider QP token query callback.
 *
 * @param [in]     qp       Verbs QP handle, or NULL when @a devx_obj is set.
 * @param [in]     devx_obj DevX QP object, or NULL when @a qp is set.
 * @param [in,out] attr     Query parameters. Only fields selected by
 *                          @a attr->field_mask should be accessed.
 *
 * @return UCS_OK on success, or an error if the operation failed.
 */
typedef ucs_status_t (*uct_ib_mlx5_ext_qp_query_func_t)(
        struct ibv_qp *qp, struct mlx5dv_devx_obj *devx_obj,
        uct_ib_mlx5_ext_qp_query_attr_t *attr);

/**
 * @brief External provider operations.
 */
typedef struct uct_ib_mlx5_ext_ops {
    char                               name[UCT_COMPONENT_NAME_MAX]; /**< Provider name */
    uct_ib_mlx5_ext_iface_flags_func_t iface_flags;                  /**< Iface flags callback */
    uct_ib_mlx5_ext_qp_query_func_t    qp_query;                     /**< QP query callback */
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
 * @param [in] ops Provider operations.
 *
 * @return UCS_OK on success, or an error if registration failed.
 */
ucs_status_t uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops);

/**
 * @brief Query iface flags from the first registered provider that supports
 * this operation.
 *
 * @param [out] flags Filled with interface capability flags.
 *
 * @return UCS_OK on success, or an error if the operation failed.
 */
ucs_status_t uct_ib_mlx5_ext_iface_flags(uint64_t *flags);

/**
 * @brief Query QP token attributes from the first registered provider that
 * supports this operation.
 *
 * @param [in]     qp       Verbs QP handle, or NULL when @a devx_obj is set.
 * @param [in]     devx_obj DevX QP object, or NULL when @a qp is set.
 * @param [in,out] attr     Query parameters and output fields.
 *
 * @return UCS_OK on success, or an error if the operation failed.
 */
ucs_status_t uct_ib_mlx5_ext_qp_query(struct ibv_qp *qp,
                                      struct mlx5dv_devx_obj *devx_obj,
                                      uct_ib_mlx5_ext_qp_query_attr_t *attr);

END_C_DECLS

#endif /* UCT_IB_MLX5_EXT_H_ */
