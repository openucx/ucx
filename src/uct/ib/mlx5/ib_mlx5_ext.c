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

UCS_LIST_HEAD(uct_ib_mlx5_ext_ops_entries);

static const uct_ib_mlx5_ext_ops_t *uct_ib_mlx5_ext_get_ops(void)
{
    uct_ib_mlx5_ext_ops_entry_t *entry;

    if (ucs_list_is_empty(&uct_ib_mlx5_ext_ops_entries)) {
        return NULL;
    }

    entry = ucs_list_head(&uct_ib_mlx5_ext_ops_entries,
                          uct_ib_mlx5_ext_ops_entry_t, list);
    return entry->ops;
}

void uct_ib_mlx5_ext_register_entry(uct_ib_mlx5_ext_ops_entry_t *entry)
{
    ucs_assert(entry->ops != NULL);
    ucs_list_add_head(&uct_ib_mlx5_ext_ops_entries, &entry->list);
}

void uct_ib_mlx5_ext_unregister_entry(uct_ib_mlx5_ext_ops_entry_t *entry)
{
    ucs_list_del(&entry->list);
}

size_t uct_ib_mlx5_ext_max_put_sgl_zcopy_count(void)
{
    const uct_ib_mlx5_ext_ops_t *ops = uct_ib_mlx5_ext_get_ops();

    if ((ops == NULL) || (ops->max_put_sgl_zcopy_count == NULL)) {
        return 0;
    }

    return ops->max_put_sgl_zcopy_count();
}

ucs_status_t
uct_ib_mlx5_ext_ep_put_sgl_zcopy(uct_ep_h ep, void * const *buffers,
                                 const size_t *lengths, uct_mem_h const *memhs,
                                 const uint64_t *remote_addrs,
                                 uct_rkey_t const *rkeys, size_t count,
                                 uct_completion_t *comp)
{
    const uct_ib_mlx5_ext_ops_t *ops = uct_ib_mlx5_ext_get_ops();

    if ((ops == NULL) || (ops->ep_put_sgl_zcopy == NULL)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ops->ep_put_sgl_zcopy(ep, buffers, lengths, memhs, remote_addrs,
                                 rkeys, count, comp);
}
