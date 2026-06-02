/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/datastruct/list.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/stubs.h>

#include "ib_mlx5_ext.h"

typedef struct uct_ib_mlx5_ext_provider {
    ucs_list_link_t       list;
    uct_ib_mlx5_ext_ops_t ops;
} uct_ib_mlx5_ext_provider_t;

UCS_LIST_HEAD(uct_ib_mlx5_ext_providers);

static ucs_status_t uct_ib_mlx5_ext_default_iface_flags(uint64_t *flags)
{
    if (ucs_likely(flags != NULL)) {
        *flags = 0;
    }

    return UCS_OK;
}

static int uct_ib_mlx5_ext_is_unsupported_op(const void *op)
{
    return (op == NULL) ||
           (op == (const void*)ucs_empty_function_return_unsupported);
}

ucs_status_t uct_ib_mlx5_ext_iface_flags(uint64_t *flags)
{
    uct_ib_mlx5_ext_provider_t *provider;

    if (ucs_unlikely(flags == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_list_for_each(provider, &uct_ib_mlx5_ext_providers, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)provider->ops.iface_flags))) {
            continue;
        }

        return provider->ops.iface_flags(flags);
    }

    return uct_ib_mlx5_ext_default_iface_flags(flags);
}

ucs_status_t uct_ib_mlx5_ext_qp_query(struct ibv_qp *qp,
                                      struct mlx5dv_devx_obj *devx_obj,
                                      uct_ib_mlx5_ext_qp_query_attr_t *attr)
{
    uct_ib_mlx5_ext_provider_t *provider;

    if (ucs_unlikely(attr == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_list_for_each(provider, &uct_ib_mlx5_ext_providers, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)provider->ops.qp_query))) {
            continue;
        }

        return provider->ops.qp_query(qp, devx_obj, attr);
    }

    return UCS_ERR_UNSUPPORTED;
}

void uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops)
{
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
    ucs_list_add_tail(&uct_ib_mlx5_ext_providers, &provider->list);

    ucs_debug("ib mlx5 ext: registered provider name=%s iface_flags=%s "
              "qp_query=%s (total=%lu)",
              provider->ops.name,
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)provider->ops.iface_flags) ?
                      "unsupported" :
                      "supported",
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)provider->ops.qp_query) ?
                      "unsupported" :
                      "supported",
              ucs_list_length(&uct_ib_mlx5_ext_providers));
}

void uct_ib_mlx5_ext_cleanup(void)
{
    uct_ib_mlx5_ext_provider_t *provider, *tmp;

    ucs_list_for_each_safe(provider, tmp, &uct_ib_mlx5_ext_providers, list) {
        ucs_list_del(&provider->list);
        ucs_free(provider);
    }

    ucs_list_head_init(&uct_ib_mlx5_ext_providers);
}
