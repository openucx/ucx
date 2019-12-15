/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_sockcm.h"


ucs_config_field_t uct_tcp_sockcm_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_tcp_sockcm_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_cm_config_table)},

  {"PRIV_DATA_LEN", "2048",
   "TCP CM private data length",
   ucs_offsetof(uct_tcp_sockcm_config_t, priv_data_len), UCS_CONFIG_TYPE_MEMUNITS},

  {NULL}
};

static ucs_status_t uct_tcp_sockcm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    uct_tcp_sockcm_t *tcp_sockcm = ucs_derived_of(cm, uct_tcp_sockcm_t);

    if (cm_attr->field_mask & UCT_CM_ATTR_FIELD_MAX_CONN_PRIV) {
        cm_attr->max_conn_priv = tcp_sockcm->priv_data_len;
    }

    return UCS_OK;
}

static uct_cm_ops_t uct_tcp_sockcm_ops = {
    .close            = UCS_CLASS_DELETE_FUNC_NAME(uct_tcp_sockcm_t),
    .cm_query         = uct_tcp_sockcm_query
};

static uct_iface_ops_t uct_tcp_sockcm_iface_ops = {
    .ep_pending_purge         = ucs_empty_function,
    .ep_disconnect            = (uct_ep_disconnect_func_t)ucs_empty_function_return_unsupported,
    .ep_destroy               = (uct_ep_destroy_func_t)ucs_empty_function_return_unsupported,
    .ep_put_short             = (uct_ep_put_short_func_t)ucs_empty_function_return_unsupported,
    .ep_put_bcopy             = (uct_ep_put_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_get_bcopy             = (uct_ep_get_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_am_short              = (uct_ep_am_short_func_t)ucs_empty_function_return_unsupported,
    .ep_am_bcopy              = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap64        = (uct_ep_atomic_cswap64_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic64_post         = (uct_ep_atomic64_post_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic64_fetch        = (uct_ep_atomic64_fetch_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap32        = (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_post         = (uct_ep_atomic32_post_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch        = (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_unsupported,
    .ep_pending_add           = (uct_ep_pending_add_func_t)ucs_empty_function_return_unsupported,
    .ep_flush                 = (uct_ep_flush_func_t)ucs_empty_function_return_unsupported,
    .ep_fence                 = (uct_ep_fence_func_t)ucs_empty_function_return_unsupported,
    .ep_check                 = (uct_ep_check_func_t)ucs_empty_function_return_unsupported,
    .ep_create                = (uct_ep_create_func_t)ucs_empty_function_return_unsupported,
    .iface_flush              = (uct_iface_flush_func_t)ucs_empty_function_return_unsupported,
    .iface_fence              = (uct_iface_fence_func_t)ucs_empty_function_return_unsupported,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (uct_iface_progress_func_t)ucs_empty_function_return_zero,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)ucs_empty_function_return_unsupported,
    .iface_event_arm          = (uct_iface_event_arm_func_t)ucs_empty_function_return_unsupported,
    .iface_close              = ucs_empty_function,
    .iface_query              = (uct_iface_query_func_t)ucs_empty_function_return_unsupported,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_unsupported,
    .iface_get_address        = (uct_iface_get_address_func_t)ucs_empty_function_return_unsupported,
    .iface_is_reachable       = (uct_iface_is_reachable_func_t)ucs_empty_function_return_zero
};

UCS_CLASS_INIT_FUNC(uct_tcp_sockcm_t, uct_component_h component,
                    uct_worker_h worker, const uct_cm_config_t *config)
{
    uct_tcp_sockcm_config_t *cm_config = ucs_derived_of(config,
                                                        uct_tcp_sockcm_config_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_cm_t, &uct_tcp_sockcm_ops,
                              &uct_tcp_sockcm_iface_ops, worker, component);

    self->priv_data_len = cm_config->priv_data_len;

    ucs_debug("created tcp_sockcm %p", self);

    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_tcp_sockcm_t)
{
}

UCS_CLASS_DEFINE(uct_tcp_sockcm_t, uct_cm_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_tcp_sockcm_t, uct_cm_t, uct_component_h,
                          uct_worker_h, const uct_cm_config_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_tcp_sockcm_t, uct_cm_t);
