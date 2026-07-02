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
#include <uct/base/uct_iface.h>
#include <ucs/type/status.h>
#include <ucs/sys/stubs.h>

BEGIN_C_DECLS

/**
 * @brief Iface query attributes field mask.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_ib_mlx5_ext_iface_query_attr_t are present.
 */
enum uct_ib_mlx5_ext_iface_query_attr_field {
    /** Enables @ref uct_ib_mlx5_ext_iface_query_attr_t::cap */
    UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS    = UCS_BIT(0),

    /** Enables @ref uct_ib_mlx5_ext_iface_query_attr_t::tx_token_len */
    UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN_LEN = UCS_BIT(1),

    /** Enables @ref uct_ib_mlx5_ext_iface_query_attr_t::rx_token_len */
    UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN_LEN = UCS_BIT(2),

    /** Enables @ref uct_ib_mlx5_ext_iface_query_attr_t::tx_token */
    UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN     = UCS_BIT(3),

    /** Enables @ref uct_ib_mlx5_ext_iface_query_attr_t::rx_token */
    UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN     = UCS_BIT(4)
};

/**
 * @brief Iface query parameters.
 */
typedef struct uct_ib_mlx5_ext_iface_query_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_ib_mlx5_ext_iface_query_attr_field. Fields not specified in
     * this mask will be ignored.
     */
    uint64_t field_mask;

    /** Interface capabilities (v2 flags) */
    struct {
        uint64_t flags; /**< Flags from @ref UCT_RESOURCE_IFACE_CAP_V2 */
    } cap;

    /** TX token length in bytes. */
    size_t tx_token_len;

    /** RX token length in bytes. */
    size_t rx_token_len;

    /** TX token input buffer, used to derive an RX token. */
    const void *tx_token;

    /** RX token output buffer derived from @ref tx_token. */
    void       *rx_token;
} uct_ib_mlx5_ext_iface_query_attr_t;

/**
 * @brief QP token query attributes field mask.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_ib_mlx5_ext_qp_query_attr_t are present.
 */
enum uct_ib_mlx5_ext_qp_query_attr_field {
    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::tx_token */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN = UCS_BIT(0),

    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::rx_token */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN = UCS_BIT(1),

    /** Enables @ref uct_ib_mlx5_ext_qp_query_attr_t::qp_num */
    UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_QP_NUM   = UCS_BIT(2)
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

    /**
     * Pointer to a caller-allocated buffer for TX token data. The buffer size
     * must be at least the TX token length returned by
     * @ref uct_ib_mlx5_ext_iface_query.
     */
    void     *tx_token;

    /**
     * Pointer to a caller-allocated buffer for RX token data. The buffer size
     * must be at least the RX token length returned by
     * @ref uct_ib_mlx5_ext_iface_query.
     */
    void     *rx_token;

    /**
     * QP number. Required as input when querying via a DevX object and
     * @a qp is NULL.
     */
    uint32_t qp_num;
} uct_ib_mlx5_ext_qp_query_attr_t;


/**
 * @brief External plugin iface query callback.
 *
 * @param [in]     iface Interface to query.
 * @param [in,out] attr  Query parameters. Only fields selected by
 *                       @a attr->field_mask should be accessed.
 *
 * @return UCS_OK on success, or an error if the operation failed.
 */
typedef ucs_status_t (*uct_ib_mlx5_ext_iface_query_func_t)(
        uct_iface_h iface, uct_ib_mlx5_ext_iface_query_attr_t *attr);

/**
 * @brief External plugin QP token query callback.
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
 * @brief External plugin maximum PUT SGL zero-copy entry count callback.
 *
 * @return Maximum number of SGL entries supported by the plugin's
 *         @ref uct_ib_mlx5_ext_ep_put_sgl_zcopy implementation, or 0 if
 *         unsupported.
 */
typedef size_t (*uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t)(void);

/**
 * @brief External plugin operations.
 */
typedef struct uct_ib_mlx5_ext_ops {
    char                                           name[UCT_COMPONENT_NAME_MAX]; /**< Plugin name */
    uct_ib_mlx5_ext_iface_query_func_t             iface_query;                  /**< Iface query callback */
    uct_ib_mlx5_ext_qp_query_func_t                qp_query;                     /**< QP query callback */
    uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t max_put_sgl_zcopy_count;      /**< Maximum PUT SGL zero-copy entry count callback */
    uct_ep_put_sgl_zcopy_func_t                    ep_put_sgl_zcopy;             /**< PUT SGL zero-copy callback */
    uct_ep_outstanding_extract_func_t              ep_outstanding_extract;       /**< Outstanding operation extract callback */
} uct_ib_mlx5_ext_ops_t;

/**
 * @brief Release mlx5 external extension.
 */
void uct_ib_mlx5_ext_cleanup(void);

/**
 * @brief Register an external plugin.
 *
 * @param [in] ops Plugin operations.
 *
 * @return UCS_OK on success, or an error if registration failed.
 */
ucs_status_t uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops);

ucs_status_t
uct_ib_mlx5_ext_iface_query(uct_iface_h iface,
                            uct_ib_mlx5_ext_iface_query_attr_t *attr);

ucs_status_t uct_ib_mlx5_ext_qp_query(struct ibv_qp *qp,
                                      struct mlx5dv_devx_obj *devx_obj,
                                      uct_ib_mlx5_ext_qp_query_attr_t *attr);

size_t uct_ib_mlx5_ext_max_put_sgl_zcopy_count(void);

ucs_status_t uct_ib_mlx5_ext_ep_put_sgl_zcopy(uct_ep_h ep,
                                              void * const *buffers,
                                              const size_t *lengths,
                                              uct_mem_h const *memhs,
                                              const uint64_t *remote_addrs,
                                              uct_rkey_t const *rkeys,
                                              const size_t *counts,
                                              const size_t *strides,
                                              size_t count,
                                              uct_completion_t *comp);

ucs_status_t uct_ib_mlx5_ext_ep_outstanding_extract(
        uct_ep_h ep, const uct_ep_outstanding_extract_params_t *params);

END_C_DECLS

#endif /* UCT_IB_MLX5_EXT_H_ */
