/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sm_iface.h"

#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucs/sys/sys.h>


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

int uct_sm_iface_is_reachable(uct_iface_t *tl_iface, const uct_device_addr_t *addr)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    return uct_sm_iface_node_guid(iface) == *(const uint64_t*)addr;
}
