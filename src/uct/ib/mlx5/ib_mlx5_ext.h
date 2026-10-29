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

struct uct_rc_mlx5_ext_ops;


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
     * Optional RC lifecycle, wireup and query hooks. Set to NULL when absent.
     * RC binds the first registered non-NULL table at interface creation and
     * uses that provider for the interface's lifetime. This does not change
     * the independent PUT/GET SGL dispatch above.
     */
    const struct uct_rc_mlx5_ext_ops *rc_ops;
} uct_ib_mlx5_ext_ops_t;

/**
 * @brief Release mlx5 external extension.
 *
 * The caller must first destroy all attached interfaces, drain callbacks and
 * stop provider timers and queued work. Serialize cleanup with registration,
 * unregistration, interface creation and plugin dispatch.
 */
void uct_ib_mlx5_ext_cleanup(void);

/**
 * @brief Unregister the first external plugin matching a name.
 *
 * Before unregistering an RC provider, destroy its attached interfaces, drain
 * callbacks and stop its timers and queued work. Serialize unregistration with
 * interface creation and plugin dispatch. Keep the provider's code, RC table
 * and referenced data alive until unregistration returns.
 *
 * @param [in] name Plugin name.
 */
void uct_ib_mlx5_ext_unregister(const char *name);

/**
 * @brief Register an external plugin.
 *
 * The registry copies this structure, including the inline name, but borrows
 * rc_ops and its referenced data. Keep the RC table immutable and valid until
 * unregistration or registry cleanup. An RC provider must use a unique name
 * that fits in name, including the terminating NUL, and register before RC
 * interfaces are opened. Serialize registration with interface creation and
 * plugin dispatch.
 *
 * Initialize unused hooks, including rc_ops, to NULL. This transport-private
 * structure requires matching rebuilt UCX and plugin consumers; RC table size
 * and version fields do not provide binary compatibility for this structure.
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
