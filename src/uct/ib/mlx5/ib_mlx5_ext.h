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

    /** 
      * TX token input buffer.
      * Valid when @ref UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN is set.
      * Caller sets this to a buffer of @ref tx_token_len bytes containing
      * the TX token received from the sender.
      * @ref UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN must be set together.
      */
    void   *tx_token;

    /**
      * RX token output buffer.
      * Valid when @ref UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN is set.
      * Caller sets this to a pre-allocated buffer of @ref rx_token_len
      * bytes; callee fills it with RX token.
      * @ref UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN must be set together.
      */
    void   *rx_token;
} uct_ib_mlx5_ext_iface_query_attr_t;

/**
 * @brief EP query attributes field mask.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_ib_mlx5_ext_ep_query_attr_t are present.
 */
enum uct_ib_mlx5_ext_ep_query_attr_field {
    /** Enables @ref uct_ib_mlx5_ext_ep_query_attr_t::tx_token */
    UCT_IB_MLX5_EXT_EP_QUERY_ATTR_FIELD_TX_TOKEN = UCS_BIT(0)
};

/**
 * @brief EP query parameters.
 */
typedef struct uct_ib_mlx5_ext_ep_query_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_ib_mlx5_ext_ep_query_attr_field. Fields not specified in this
     * mask will be ignored.
     */
    uint64_t field_mask;

    /**
     * Pointer to a caller-allocated buffer for TX token data. The buffer size
     * must be at least the TX token length returned by
     * @ref uct_ib_mlx5_ext_iface_query.
     */
    void     *tx_token;
} uct_ib_mlx5_ext_ep_query_attr_t;

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
 * @brief External plugin EP query callback.
 *
 * @param [in]     ep    Endpoint to query.
 * @param [in,out] attr  Query parameters. Only fields selected by
 *                       @a attr->field_mask should be accessed.
 *
 * @return UCS_OK on success, or an error if the operation failed.
 */
typedef ucs_status_t (*uct_ib_mlx5_ext_ep_query_func_t)(
        uct_ep_h ep, uct_ib_mlx5_ext_ep_query_attr_t *attr);

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
    uct_ib_mlx5_ext_ep_query_func_t                ep_query;                     /**< EP query callback */
    uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t max_put_sgl_zcopy_count;      /**< Maximum PUT SGL zero-copy entry count callback */
    uct_ep_put_sgl_zcopy_func_t                    ep_put_sgl_zcopy;             /**< PUT SGL zero-copy callback */
    uct_ep_outstanding_purge_func_t                ep_outstanding_purge;         /**< Outstanding operation purge callback */
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

ucs_status_t
uct_ib_mlx5_ext_ep_query(uct_ep_h ep, uct_ib_mlx5_ext_ep_query_attr_t *attr);

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

END_C_DECLS

#endif /* UCT_IB_MLX5_EXT_H_ */
