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
    UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS = UCS_BIT(0)
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
} uct_ib_mlx5_ext_iface_query_attr_t;

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
    uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t max_put_sgl_zcopy_count;      /**< Maximum PUT SGL zero-copy entry count callback */
    uct_ep_put_sgl_zcopy_func_t                    ep_put_sgl_zcopy;             /**< PUT SGL zero-copy callback */
    uct_ep_outstanding_purge_func_t                ep_outstanding_purge;         /**< Outstanding operation purge callback */
} uct_ib_mlx5_ext_ops_t;

/**
 * @brief Release mlx5 external extension.
 */
void uct_ib_mlx5_ext_cleanup(void);

/**
 * @brief Unregister the first external plugin matching a name.
 *
 * @param [in] name Plugin name.
 */
void uct_ib_mlx5_ext_unregister(const char *name);

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

ucs_status_t uct_ib_mlx5_ext_ep_outstanding_purge(
        uct_ep_h ep, const uct_ep_outstanding_purge_params_t *params);

END_C_DECLS

#endif /* UCT_IB_MLX5_EXT_H_ */
