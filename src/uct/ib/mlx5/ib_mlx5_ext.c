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
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/stubs.h>
#include <ucs/type/spinlock.h>

typedef struct uct_ib_mlx5_ext_plugin {
    ucs_list_link_t       list;
    uct_ib_mlx5_ext_ops_t ops;
} uct_ib_mlx5_ext_plugin_t;

UCS_LIST_HEAD(uct_ib_mlx5_ext_plugins);

static ucs_spinlock_t uct_ib_mlx5_ext_lock;
static int            uct_ib_mlx5_ext_initialized;

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

static ucs_status_t uct_ib_mlx5_ext_qp_query_check_param(
        struct ibv_qp *qp, struct mlx5dv_devx_obj *devx_obj,
        const uct_ib_mlx5_ext_qp_query_attr_t *attr)
{
    const uint64_t token_mask = UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN |
                                UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN;

    if (attr->field_mask & UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_TX_TOKEN) {
        if (attr->tx_token == NULL) {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    if (attr->field_mask & UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_RX_TOKEN) {
        if (attr->rx_token == NULL) {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    if (attr->field_mask & token_mask) {
        if ((qp == NULL) && (devx_obj == NULL)) {
            return UCS_ERR_INVALID_PARAM;
        }

        if ((qp == NULL) && (devx_obj != NULL) &&
            !(attr->field_mask & UCT_IB_MLX5_EXT_QP_QUERY_ATTR_FIELD_QP_NUM)) {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_ext_iface_query(uct_iface_h iface,
                            uct_ib_mlx5_ext_iface_query_attr_t *attr)
{
    uct_ib_mlx5_ext_plugin_t *plugin;
    ucs_status_t status;

    if (ucs_unlikely(attr == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);

    ucs_list_for_each(plugin, &uct_ib_mlx5_ext_plugins, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)plugin->ops.iface_query))) {
            continue;
        }

        status = plugin->ops.iface_query(iface, attr);
        ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
        return status;
    }

    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
    return uct_ib_mlx5_ext_iface_query_default(attr);
}

ucs_status_t uct_ib_mlx5_ext_qp_query(struct ibv_qp *qp,
                                      struct mlx5dv_devx_obj *devx_obj,
                                      uct_ib_mlx5_ext_qp_query_attr_t *attr)
{
    uct_ib_mlx5_ext_plugin_t *plugin;
    ucs_status_t status;

    if (ucs_unlikely(attr == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_ib_mlx5_ext_qp_query_check_param(qp, devx_obj, attr);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);

    ucs_list_for_each(plugin, &uct_ib_mlx5_ext_plugins, list) {
        if (ucs_unlikely(uct_ib_mlx5_ext_is_unsupported_op(
                    (const void*)plugin->ops.qp_query))) {
            continue;
        }

        status = plugin->ops.qp_query(qp, devx_obj, attr);
        ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
        return status;
    }

    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);
    return UCS_ERR_UNSUPPORTED;
}

void uct_ib_mlx5_ext_init(void)
{
    ucs_status_t status;

    if (uct_ib_mlx5_ext_initialized) {
        return;
    }

    status = ucs_spinlock_init(&uct_ib_mlx5_ext_lock, 0);
    if (status != UCS_OK) {
        ucs_fatal("failed to initialize mlx5 ext lock: %s",
                  ucs_status_string(status));
    }

    uct_ib_mlx5_ext_initialized = 1;
}

void uct_ib_mlx5_ext_cleanup(void)
{
    uct_ib_mlx5_ext_plugin_t *plugin, *tmp;

    if (!uct_ib_mlx5_ext_initialized) {
        return;
    }

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);

    ucs_list_for_each_safe(plugin, tmp, &uct_ib_mlx5_ext_plugins, list) {
        ucs_list_del(&plugin->list);
        ucs_free(plugin);
    }

    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);

    ucs_spinlock_destroy(&uct_ib_mlx5_ext_lock);
    uct_ib_mlx5_ext_initialized = 0;
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

    ucs_spin_lock(&uct_ib_mlx5_ext_lock);
    ucs_list_add_tail(&uct_ib_mlx5_ext_plugins, &plugin->list);
    num_plugins = ucs_list_length(&uct_ib_mlx5_ext_plugins);
    ucs_spin_unlock(&uct_ib_mlx5_ext_lock);

    ucs_debug("ib mlx5 ext: registered plugin name=%s iface_query=%s "
              "qp_query=%s (total=%u)",
              plugin->ops.name,
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)plugin->ops.iface_query) ?
                      "unsupported" :
                      "supported",
              uct_ib_mlx5_ext_is_unsupported_op(
                      (const void*)plugin->ops.qp_query) ?
                      "unsupported" :
                      "supported",
              num_plugins);
    return UCS_OK;
}
