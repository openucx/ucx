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
#include <ucs/type/spinlock.h>

#include "ib_mlx5_ext.h"

typedef struct uct_ib_mlx5_ext_provider {
    ucs_list_link_t       list;
    uct_ib_mlx5_ext_ops_t ops;
} uct_ib_mlx5_ext_provider_t;

UCS_LIST_HEAD(uct_ib_mlx5_ext_providers);

static ucs_spinlock_t uct_ib_mlx5_ext_lock;

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
    ucs_status_t status;

    if (ucs_unlikely(flags == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);

    ucs_list_for_each(provider, &uct_ib_mlx5_ext_providers, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)provider->ops.iface_flags))) {
            continue;
        }

        status = provider->ops.iface_flags(flags);
        ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
        return status;
    }

    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
    return uct_ib_mlx5_ext_default_iface_flags(flags);
}

ucs_status_t uct_ib_mlx5_ext_qp_query(struct ibv_qp *qp,
                                      struct mlx5dv_devx_obj *devx_obj,
                                      uct_ib_mlx5_ext_qp_query_attr_t *attr)
{
    uct_ib_mlx5_ext_provider_t *provider;
    ucs_status_t status;

    if (ucs_unlikely(attr == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);

    ucs_list_for_each(provider, &uct_ib_mlx5_ext_providers, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)provider->ops.qp_query))) {
            continue;
        }

        status = provider->ops.qp_query(qp, devx_obj, attr);
        ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
        return status;
    }

    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
    return UCS_ERR_UNSUPPORTED;
}

void uct_ib_mlx5_ext_init(void)
{
    ucs_status_t status;

    status = ucs_spinlock_init(&uct_ib_mlx5_ext_lock, 0);
    if (status != UCS_OK) {
        ucs_fatal("failed to initialize mlx5 ext lock: %s",
                  ucs_status_string(status));
    }
}

void uct_ib_mlx5_ext_cleanup(void)
{
    uct_ib_mlx5_ext_provider_t *provider, *tmp;

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);

    ucs_list_for_each_safe(provider, tmp, &uct_ib_mlx5_ext_providers, list) {
        ucs_list_del(&provider->list);
        ucs_free(provider);
    }

    ucs_list_head_init(&uct_ib_mlx5_ext_providers);
    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);

    ucs_spinlock_destroy(&uct_ib_mlx5_ext_lock);
}

void uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops)
{
    uct_ib_mlx5_ext_provider_t *provider;
    unsigned num_providers;

    if (ucs_unlikely(ops == NULL)) {
        ucs_debug("ib mlx5 ext: ignored NULL provider");
        return;
    }

    provider = ucs_malloc(sizeof(*provider), "mlx5_ext_provider");
    if (ucs_unlikely(provider == NULL)) {
        ucs_error("ib mlx5 ext: failed to allocate provider entry for %s",
                  ops->name);
        return;
    }

    provider->ops = *ops;

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);
    ucs_list_add_tail(&uct_ib_mlx5_ext_providers, &provider->list);
    num_providers = ucs_list_length(&uct_ib_mlx5_ext_providers);
    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);

    ucs_debug("ib mlx5 ext: registered provider name=%s iface_flags=%s "
              "qp_query=%s (total=%u)",
              provider->ops.name,
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)provider->ops.iface_flags) ?
                      "unsupported" :
                      "supported",
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)provider->ops.qp_query) ?
                      "unsupported" :
                      "supported",
              num_providers);
}
