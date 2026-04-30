/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/stubs.h>

#include "ib_mlx5_ext.h"



uct_ib_mlx5_ext_ops_t uct_ib_mlx5_ext_ops = {
    .iface_flags = (uct_ib_mlx5_ext_iface_flags_func_t)ucs_empty_function_return_unsupported,
    .qp_query    = (uct_ib_mlx5_ext_qp_query_func_t)ucs_empty_function_return_unsupported,
};


void uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops)
{
    ucs_assertv(ops->iface_flags != NULL, "ext_resgiter: iface_flags function is NULL");
    ucs_assertv(ops->qp_query != NULL, "ext_resgiter: qp_query function is NULL");
    uct_ib_mlx5_ext_ops = *ops;
    ucs_info("ib mlx5: registered external ops successfully");
}
