/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/datastruct/list.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/stubs.h>

#include "ib_mlx5_ext.h"

UCS_LIST_HEAD(uct_ib_mlx5_ext_providers);

typedef struct uct_ib_mlx5_ext_provider {
    ucs_list_link_t       list;
    uct_ib_mlx5_ext_ops_t ops;
} uct_ib_mlx5_ext_provider_t;

static ucs_status_t uct_ib_mlx5_ext_default_iface_flags(uint64_t *flags)
{
    if (ucs_likely(flags != NULL)) {
        *flags = 0;
    }

    return UCS_OK;
}

static uct_ib_mlx5_ext_ops_t uct_ib_mlx5_ext_ops_default = {
    .name        = "default",
    .iface_flags = uct_ib_mlx5_ext_default_iface_flags,
    .qp_query    = (uct_ib_mlx5_ext_qp_query_func_t)ucs_empty_function_return_unsupported,
};

static uct_ib_mlx5_ext_ops_t uct_ib_mlx5_ext_ops_current =
        uct_ib_mlx5_ext_ops_default;

static int uct_ib_mlx5_ext_is_unsupported_op(const void *op)
{
    return (op == NULL) ||
           (op == (const void*)ucs_empty_function_return_unsupported);
}

ucs_status_t uct_ib_mlx5_ext_iface_flags(uint64_t *flags)
{
    if (ucs_unlikely(flags == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return uct_ib_mlx5_ext_ops_current.iface_flags(flags);
}

ucs_status_t uct_ib_mlx5_ext_qp_query(struct ibv_qp *qp,
                                      struct mlx5dv_devx_obj *devx_obj,
                                      uct_ib_mlx5_ext_qp_query_attr_t *attr)
{
    if (ucs_unlikely(attr == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return uct_ib_mlx5_ext_ops_current.qp_query(qp, devx_obj, attr);
}

void uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops)
{
    int updated = 0;
    uct_ib_mlx5_ext_provider_t *provider;

    if (ucs_unlikely(ops == NULL)) {
        ucs_debug("ib mlx5 ext: ignored NULL provider");
        return;
    }

    provider = ucs_malloc(sizeof(*provider), "mlx5_ext_provider");
    if (ucs_unlikely(provider == NULL)) {
        ucs_debug("ib mlx5 ext: failed to allocate provider entry for %s",
                  ops->name);
        return;
    }

    provider->ops = *ops;

    ucs_list_add_head(&uct_ib_mlx5_ext_providers, &provider->list);

    if (!(uct_ib_mlx5_ext_is_unsupported_op(ops->iface_flags))) {
        uct_ib_mlx5_ext_ops_current.iface_flags = ops->iface_flags;
        updated                                 = 1;
    }

    if (!(uct_ib_mlx5_ext_is_unsupported_op(ops->qp_query))) {
        uct_ib_mlx5_ext_ops_current.qp_query = ops->qp_query;
        updated                              = 1;
    }

    if (updated) {
        ucs_strncpy(uct_ib_mlx5_ext_ops_current.name, ops->name,
                    sizeof(uct_ib_mlx5_ext_ops_current.name));
        ucs_debug("ib mlx5 ext: updated provider name=%s iface_flags=%s "
                  "qp_query=%s",
                  ops->name,
                  uct_ib_mlx5_ext_is_unsupported_op(ops->iface_flags) ?
                          "unsupported" :
                          "supported",
                  uct_ib_mlx5_ext_is_unsupported_op(ops->qp_query) ?
                          "unsupported" :
                          "supported");
    }
}