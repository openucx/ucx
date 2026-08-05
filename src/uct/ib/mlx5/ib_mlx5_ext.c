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

static ucs_status_t
uct_ib_mlx5_ext_iface_query_default(uct_ib_mlx5_ext_iface_query_attr_t *attr)
{
    const uint64_t token_len_mask =
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN_LEN |
            UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN_LEN;

    if (attr->field_mask & UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_CAP_FLAGS) {
        attr->cap.flags = 0;
    }

    if (attr->field_mask & token_len_mask) {
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

static int uct_ib_mlx5_ext_is_unsupported_op(const void *op)
{
    return (op == NULL) ||
           (op == (const void*)ucs_empty_function_return_unsupported);
}

static ucs_status_t
uct_ib_mlx5_ext_iface_query_check_param(uct_ib_mlx5_ext_iface_query_attr_t *attr)
{
    if (attr == NULL) {
        ucs_error("ib mlx5 ext: iface query attribute is NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    if ((attr->field_mask & UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN) &&
        (attr->tx_token == NULL)) {
        ucs_error("ib mlx5 ext: tx token is NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    if ((attr->field_mask & UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN) &&
        (attr->rx_token == NULL)) {
        ucs_error("ib mlx5 ext: rx token is NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    if (!!(attr->field_mask &
           UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_TX_TOKEN) !=
        !!(attr->field_mask &
           UCT_IB_MLX5_EXT_IFACE_QUERY_ATTR_FIELD_RX_TOKEN)) {
        ucs_error("ib mlx5 ext: tx token and rx token must be set together");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t
uct_ib_mlx5_ext_ep_query_check_param(uct_ib_mlx5_ext_ep_query_attr_t *attr)
{
    if (attr == NULL) {
        ucs_error("ib mlx5 ext: ep query attribute is NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    if ((attr->field_mask & UCT_IB_MLX5_EXT_EP_QUERY_ATTR_FIELD_TX_TOKEN) &&
        (attr->tx_token == NULL)) {
        ucs_error("ib mlx5 ext: tx token is NULL");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_ext_iface_query(uct_iface_h iface,
                            uct_ib_mlx5_ext_iface_query_attr_t *attr)
{
    uct_ib_mlx5_ext_plugin_t *plugin;

    if (ucs_unlikely(uct_ib_mlx5_ext_iface_query_check_param(attr) != UCS_OK)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_list_for_each(plugin, &uct_ib_mlx5_ext_plugins, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)plugin->ops.iface_query))) {
            continue;
        }

        return plugin->ops.iface_query(iface, attr);
    }

    return uct_ib_mlx5_ext_iface_query_default(attr);
}

ucs_status_t
uct_ib_mlx5_ext_ep_query(uct_ep_h ep, uct_ib_mlx5_ext_ep_query_attr_t *attr)
{
    uct_ib_mlx5_ext_plugin_t *plugin;

    if (ucs_unlikely(uct_ib_mlx5_ext_ep_query_check_param(attr) != UCS_OK)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_list_for_each(plugin, &uct_ib_mlx5_ext_plugins, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)plugin->ops.ep_query))) {
            continue;
        }

        return plugin->ops.ep_query(ep, attr);
    }

    return UCS_ERR_UNSUPPORTED;
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

    ucs_debug("ib mlx5 ext: registered plugin name=%s iface_query=%s "
              "ep_query=%s put_sgl_zcopy=%s outstanding_purge=%s "
              "(total=%u)",
              plugin->ops.name,
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)plugin->ops.iface_query) ?
                      "unsupported" :
                      "supported",
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)plugin->ops.ep_query) ?
                      "unsupported" :
                      "supported",
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)plugin->ops.ep_put_sgl_zcopy) ?
                      "unsupported" :
                      "supported",
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)plugin->ops.ep_outstanding_purge) ?
                      "unsupported" :
                      "supported",
              num_plugins);
    return UCS_OK;
}
