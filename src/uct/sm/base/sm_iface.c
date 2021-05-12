/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "sm_iface.h"

#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/arch/cpu.h>
#include <ucs/type/init_once.h>


ucs_config_field_t uct_sm_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_sm_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"BW", "12179MBs",
     "Effective memory bandwidth",
     ucs_offsetof(uct_sm_iface_config_t, bandwidth), UCS_CONFIG_TYPE_BW},

    {NULL}
};

ucs_status_t
uct_sm_base_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p)
{
    return uct_single_device_resource(md, UCT_SM_DEVICE_NAME,
                                      UCT_DEVICE_TYPE_SHM,
                                      UCS_SYS_DEVICE_ID_UNKNOWN, tl_devices_p,
                                      num_tl_devices_p);
}

ucs_status_t
uct_sm_iface_get_device_address(uct_iface_t *tl_iface, uct_device_addr_t *addr)
{
    uct_iface_get_local_address((uct_iface_local_addr_ns_t*)addr,
                                UCS_SYS_NS_TYPE_IPC);
    return UCS_OK;
}

int uct_sm_iface_is_reachable(const uct_iface_h tl_iface,
                              const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr)
{
    return uct_iface_local_is_reachable((uct_iface_local_addr_ns_t*)dev_addr,
                                        UCS_SYS_NS_TYPE_IPC);
}

ucs_status_t uct_sm_iface_fence(uct_iface_t *tl_iface, unsigned flags)
{
    ucs_memory_cpu_fence();
    UCT_TL_IFACE_STAT_FENCE(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

ucs_status_t uct_sm_ep_fence(uct_ep_t *tl_ep, unsigned flags)
{
    ucs_memory_cpu_fence();
    UCT_TL_EP_STAT_FENCE(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_OK;
}

size_t uct_sm_iface_get_device_addr_len()
{
    return ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_IPC) ?
                   sizeof(uct_iface_local_addr_base_t) :
                   sizeof(uct_iface_local_addr_ns_t);
}

UCS_CLASS_INIT_FUNC(uct_sm_iface_t, uct_iface_ops_t *ops,
                    uct_iface_internal_ops_t *internal_ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config)
{
    uct_sm_iface_config_t *sm_config = ucs_derived_of(tl_config,
                                                      uct_sm_iface_config_t);

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");
    if (!(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE)) {
        ucs_error("only UCT_IFACE_OPEN_MODE_DEVICE is supported");
        return UCS_ERR_UNSUPPORTED;
    }

    UCS_CLASS_CALL_SUPER_INIT(
            uct_base_iface_t, ops, internal_ops, md, worker, params,
            tl_config UCS_STATS_ARG(
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                            params->stats_root :
                            NULL) UCS_STATS_ARG(params->mode.device.dev_name));

    self->config.bandwidth = sm_config->bandwidth;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sm_iface_t)
{
}

UCS_CLASS_DEFINE(uct_sm_iface_t, uct_base_iface_t);
