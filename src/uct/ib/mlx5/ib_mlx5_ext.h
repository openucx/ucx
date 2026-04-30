/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_MLX5_EXT_H_
#define UCT_IB_MLX5_EXT_H_

#include <uct/api/uct_def.h>
#include <uct/base/uct_iface.h>
#include <ucs/datastruct/list.h>

BEGIN_C_DECLS

typedef size_t (*uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t)(void);

typedef struct uct_ib_mlx5_ext_ops {
    uct_ib_mlx5_ext_max_put_sgl_zcopy_count_func_t max_put_sgl_zcopy_count;
    uct_ep_put_sgl_zcopy_func_t                    ep_put_sgl_zcopy;
} uct_ib_mlx5_ext_ops_t;

typedef struct uct_ib_mlx5_ext_provider {
    ucs_list_link_t             list;
    const uct_ib_mlx5_ext_ops_t *ops;
    int                         registered;
} uct_ib_mlx5_ext_provider_t;

void uct_ib_mlx5_ext_register_provider(uct_ib_mlx5_ext_provider_t *provider);

void uct_ib_mlx5_ext_unregister_provider(uct_ib_mlx5_ext_provider_t *provider);

size_t uct_ib_mlx5_ext_max_put_sgl_zcopy_count(void);

ucs_status_t uct_ib_mlx5_ext_ep_put_sgl_zcopy(uct_ep_h ep,
                                              void * const *buffers,
                                              const size_t *lengths,
                                              uct_mem_h const *memhs,
                                              const uint64_t *remote_addrs,
                                              uct_rkey_t const *rkeys,
                                              size_t count,
                                              uct_completion_t *comp);

END_C_DECLS

#endif
