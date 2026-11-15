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

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/status.h>
#include <ucs/sys/stubs.h>

BEGIN_C_DECLS

/**
 * @brief External plugin maximum PUT SGL zero-copy entry count callback.
 *
 * @return Maximum number of SGL entries supported by the plugin's
 *         @ref uct_ib_mlx5_ext_ep_put_sgl_zcopy implementation, or 0 if
 *         unsupported.
 */
typedef size_t (*uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t)(void);

/**
 * @brief External plugin maximum GET SGL zero-copy entry count callback.
 *
 * @return Maximum number of SGL entries supported by the plugin's
 *         @ref uct_ib_mlx5_ext_ep_get_sgl_zcopy implementation, or 0 if
 *         unsupported.
 */
typedef size_t (*uct_ib_mlx5_ext_max_get_sgl_zcopy_count_func_t)(void);


/* RC-specific provider hooks. */

enum {
    /**
     * Require successful provider attachment at interface creation. If
     * iface_init fails, abort interface creation. If unset, normal RC
     * continues without the provider when attachment fails. This policy does
     * not suppress ordinary RC errors or endpoint hook errors.
     */
    UCT_RC_MLX5_EXT_FLAG_REQUIRE_IFACE_ATTACH = UCS_BIT(0)
};


/**
 * @brief RC mlx5 operations implemented by an external plugin provider.
 */
typedef struct uct_rc_mlx5_ext_ops {
    /**
     * Interface attachment policy flags; see
     * @ref UCT_RC_MLX5_EXT_FLAG_REQUIRE_IFACE_ATTACH.
     */
    uint64_t        flags;

    /**
     * @brief Initialize provider state on an existing RC interface.
     *
     * @param [in]  iface       Existing RC interface.
     * @param [out] iface_ctx_p Provider context, published only on success.
     * @return UCS_OK on success, or an error if initialization failed.
     */
    ucs_status_t (*iface_init)(uct_iface_h iface, void **iface_ctx_p);

    /**
     * @brief Release provider interface state after endpoint cleanup.
     *
     * @param [in] iface_ctx Provider context from iface_init.
     */
    void (*iface_cleanup)(void *iface_ctx);

    /**
     * @brief Initialize provider state on an existing RC endpoint.
     *
     * @param [in]  ep        Existing RC endpoint.
     * @param [in]  iface_ctx Provider context from iface_init.
     * @param [out] ep_ctx_p  Provider context, published only on success.
     * @return UCS_OK on success, or an error if initialization failed.
     */
    ucs_status_t (*ep_init)(uct_ep_h ep, void *iface_ctx, void **ep_ctx_p);

    /**
     * @brief Release provider endpoint state.
     *
     * @param [in] ep_ctx Provider context from ep_init.
     */
    void (*ep_cleanup)(void *ep_ctx);

    /**
     * @brief Return the latest available cached endpoint bandwidth.
     *
     * This optional hook requires the endpoint bandwidth-query attribute
     * extension. It may be omitted when that extension is unavailable.
     *
     * @param [in]     ep_ctx  Provider context from ep_init.
     * @param [in,out] ep_attr Requested attributes and provider-owned outputs.
     * @return UCS_OK if the requested bandwidth was filled.
     */
    ucs_status_t (*ep_query)(void *ep_ctx, uct_ep_attr_t *ep_attr);
} uct_rc_mlx5_ext_ops_t;


/**
 * @brief External plugin operations.
 */
typedef struct uct_ib_mlx5_ext_ops {
    char                                           name[UCT_COMPONENT_NAME_MAX]; /**< Plugin name */
    uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t max_put_sgl_zcopy_count;      /**< Maximum PUT SGL zero-copy entry count callback */
    uct_ep_put_sgl_zcopy_func_t                    ep_put_sgl_zcopy;             /**< PUT SGL zero-copy callback */
    uct_ib_mlx5_ext_max_get_sgl_zcopy_count_func_t max_get_sgl_zcopy_count;      /**< Maximum GET SGL zero-copy entry count callback */
    uct_ep_get_sgl_zcopy_func_t                    ep_get_sgl_zcopy;             /**< GET SGL zero-copy callback */

    /**
     * Table of RC-specific operations implemented by the plugin provider,
     * used only when the RC transport is enabled.
     */
    const uct_rc_mlx5_ext_ops_t *rc_ops;
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

size_t uct_ib_mlx5_ext_max_get_sgl_zcopy_count(void);

ucs_status_t uct_ib_mlx5_ext_ep_get_sgl_zcopy(uct_ep_h ep,
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
