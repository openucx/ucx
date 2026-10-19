/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ib_mlx5_ext.h"

#include <ucs/datastruct/list.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/stubs.h>

typedef struct uct_ib_mlx5_ext_plugin {
    ucs_list_link_t       list;
    uct_ib_mlx5_ext_ops_t ops;
} uct_ib_mlx5_ext_plugin_t;

UCS_LIST_HEAD(uct_ib_mlx5_ext_plugins);

static int uct_ib_mlx5_ext_is_unsupported_op(const void *op)
{
    return (op == NULL) ||
           (op == (const void*)ucs_empty_function_return_unsupported);
}

size_t uct_ib_mlx5_ext_max_put_sgl_zcopy_count(void)
{
    uct_ib_mlx5_ext_plugin_t *plugin;

    ucs_list_for_each(plugin, &uct_ib_mlx5_ext_plugins, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)plugin->ops.max_put_sgl_zcopy_count))) {
            continue;
        }

        return plugin->ops.max_put_sgl_zcopy_count();
    }

    return 0;
}

ucs_status_t
uct_ib_mlx5_ext_ep_put_sgl_zcopy(uct_ep_h ep, void * const *buffers,
                                 const size_t *lengths, uct_mem_h const *memhs,
                                 const uint64_t *remote_addrs,
                                 uct_rkey_t const *rkeys, const size_t *counts,
                                 const size_t *strides, size_t count,
                                 uct_completion_t *comp)
{
    uct_ib_mlx5_ext_plugin_t *plugin;

    ucs_list_for_each(plugin, &uct_ib_mlx5_ext_plugins, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)plugin->ops.ep_put_sgl_zcopy))) {
            continue;
        }

        return plugin->ops.ep_put_sgl_zcopy(ep, buffers, lengths, memhs,
                                            remote_addrs, rkeys, counts,
                                            strides, count, comp);
    }

    return UCS_ERR_UNSUPPORTED;
}

void uct_ib_mlx5_ext_cleanup(void)
{
    uct_ib_mlx5_ext_plugin_t *plugin, *tmp;

    ucs_list_for_each_safe(plugin, tmp, &uct_ib_mlx5_ext_plugins, list) {
        ucs_list_del(&plugin->list);
        ucs_free(plugin);
    }
}

void uct_ib_mlx5_ext_unregister(const char *name)
{
    uct_ib_mlx5_ext_plugin_t *plugin, *tmp;

    if (name == NULL) {
        return;
    }

    ucs_list_for_each_safe(plugin, tmp, &uct_ib_mlx5_ext_plugins, list) {
        if (!strncmp(plugin->ops.name, name, UCT_COMPONENT_NAME_MAX)) {
            ucs_list_del(&plugin->list);
            ucs_free(plugin);
            return;
        }
    }
}

ucs_status_t uct_ib_mlx5_ext_register(const uct_ib_mlx5_ext_ops_t *ops)
{
    uct_ib_mlx5_ext_plugin_t *plugin;
    unsigned num_plugins;

    if (ucs_unlikely(ops == NULL)) {
        ucs_warn("ib mlx5 ext: ignored NULL plugin");
        return UCS_ERR_INVALID_PARAM;
    }

    plugin = ucs_malloc(sizeof(*plugin), "mlx5_ext_plugin");
    if (ucs_unlikely(plugin == NULL)) {
        ucs_error("ib mlx5 ext: failed to allocate plugin entry for %.*s",
                  UCT_COMPONENT_NAME_MAX, ops->name);
        return UCS_ERR_NO_MEMORY;
    }

    plugin->ops                                  = *ops;
    plugin->ops.name[UCT_COMPONENT_NAME_MAX - 1] = '\0';

    ucs_list_add_tail(&uct_ib_mlx5_ext_plugins, &plugin->list);
    num_plugins = ucs_list_length(&uct_ib_mlx5_ext_plugins);

    ucs_debug("ib mlx5 ext: registered plugin name=%s put_sgl_zcopy=%s "
              "(total=%u)",
              plugin->ops.name,
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)plugin->ops.ep_put_sgl_zcopy) ?
                      "unsupported" :
                      "supported",
              num_plugins);
    return UCS_OK;
}
