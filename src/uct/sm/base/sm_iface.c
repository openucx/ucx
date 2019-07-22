/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sm_iface.h"

#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/arch/cpu.h>

ucs_config_field_t uct_sm_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_sm_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"BW", "12179MBs",
     "Effective memory bandwidth",
     ucs_offsetof(uct_sm_iface_config_t, bandwidth), UCS_CONFIG_TYPE_BW},

    {NULL}
};

static uint64_t uct_sm_iface_node_guid(uct_base_iface_t *iface)
{
    /* The address should be different for different mm 'devices' so that
     * they won't seem reachable one to another. Their 'name' will create the
     * uniqueness in the address */
    return ucs_machine_guid() *
           ucs_string_to_id(iface->md->component->name);
}

ucs_status_t uct_sm_iface_get_device_address(uct_iface_t *tl_iface,
                                             uct_device_addr_t *addr)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    *(uint64_t*)addr = uct_sm_iface_node_guid(iface);
    return UCS_OK;
}

int uct_sm_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    return uct_sm_iface_node_guid(iface) == *(const uint64_t*)dev_addr;
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

UCS_CLASS_INIT_FUNC(uct_sm_iface_t, uct_iface_ops_t *ops, uct_md_h md,
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

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, ops, md, worker, params,
                              tl_config
                              UCS_STATS_ARG((params->field_mask &
                                             UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                                            params->stats_root : NULL)
                              UCS_STATS_ARG(params->mode.device.dev_name));

    self->config.bandwidth = sm_config->bandwidth;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sm_iface_t)
{
}

UCS_CLASS_DEFINE(uct_sm_iface_t, uct_base_iface_t);


