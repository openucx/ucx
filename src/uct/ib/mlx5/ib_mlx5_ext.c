/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ib_mlx5_ext.h"

#include <ucs/debug/assert.h>

UCS_LIST_HEAD(uct_ib_mlx5_ext_providers_list);

static const uct_ib_mlx5_ext_ops_t *uct_ib_mlx5_ext_published_ops;

static void uct_ib_mlx5_ext_republish(void)
{
    uct_ib_mlx5_ext_provider_t *iter;
    uct_ib_mlx5_ext_provider_t *chosen = NULL;

    ucs_list_for_each(iter, &uct_ib_mlx5_ext_providers_list, list) {
        if ((iter->ops != NULL) && (iter->ops->ep_put_sgl_zcopy != NULL)) {
            chosen = iter;
            break;
        }
    }

    uct_ib_mlx5_ext_published_ops = (chosen != NULL) ? chosen->ops : NULL;
}

void uct_ib_mlx5_ext_register_provider(uct_ib_mlx5_ext_provider_t *provider)
{
    ucs_assert(provider->ops != NULL);

    if (provider->registered) {
        return;
    }

    ucs_list_add_tail(&uct_ib_mlx5_ext_providers_list, &provider->list);
    provider->registered = 1;
    uct_ib_mlx5_ext_republish();
}

void uct_ib_mlx5_ext_unregister_provider(uct_ib_mlx5_ext_provider_t *provider)
{
    if (!provider->registered) {
        return;
    }

    ucs_list_del(&provider->list);
    provider->registered = 0;
    uct_ib_mlx5_ext_republish();
}

static const uct_ib_mlx5_ext_ops_t *uct_ib_mlx5_ext_active_ops(void)
{
    return uct_ib_mlx5_ext_published_ops;
}

size_t uct_ib_mlx5_ext_max_put_sgl_zcopy_count(void)
{
    const uct_ib_mlx5_ext_ops_t *ops = uct_ib_mlx5_ext_active_ops();

    if ((ops == NULL) || (ops->max_put_sgl_zcopy_count == NULL)) {
        return 0;
    }

    return ops->max_put_sgl_zcopy_count();
}

ucs_status_t
uct_ib_mlx5_ext_ep_put_sgl_zcopy(uct_ep_h ep,
                                 void * const *buffers,
                                 const size_t *lengths,
                                 uct_mem_h const *memhs,
                                 const uint64_t *remote_addrs,
                                 uct_rkey_t const *rkeys,
                                 size_t count,
                                 uct_completion_t *comp)
{
    const uct_ib_mlx5_ext_ops_t *ops = uct_ib_mlx5_ext_active_ops();

    if ((ops == NULL) || (ops->ep_put_sgl_zcopy == NULL)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ops->ep_put_sgl_zcopy(ep, buffers, lengths, memhs, remote_addrs,
                                 rkeys, count, comp);
}
