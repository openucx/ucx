/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_context.h"
#include "ib_verbs.h"

#include <uct/tl/context.h>
#include <ucs/type/component.h>
#include <ucs/debug/memtrack.h>


ucs_status_t uct_ib_query_resources(uct_context_h context, unsigned flags,
                                    uct_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    uct_ib_context_t *ibctx = ucs_component_get(context, ib, uct_ib_context_t);
    uct_resource_desc_t *resources, *rsc;
    ucs_status_t status;
    unsigned num_resources;
    unsigned resource_index;
    uct_ib_device_t *dev;
    unsigned dev_index;
    uint8_t port_num;

    /* First pass: count ports */
    num_resources = 0;
    for (dev_index = 0; dev_index < ibctx->num_devices; ++dev_index) {
        dev = ibctx->devices[dev_index];
        for (port_num = dev->first_port; port_num < dev->first_port + dev->num_ports;
                        ++port_num)
        {
            if (uct_ib_device_port_check(dev, port_num, flags)) {
                ++num_resources;
            }
        }
    }

    /* Allocate resources array */
    resources = ucs_calloc(num_resources, sizeof(uct_resource_desc_t), "resource desc");
    if (resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* Second pass: fill port information */
    resource_index = 0;
    for (dev_index = 0; dev_index < ibctx->num_devices; ++dev_index) {
        dev = ibctx->devices[dev_index];
        for (port_num = dev->first_port; port_num < dev->first_port + dev->num_ports;
                        ++port_num)
        {
            if (uct_ib_device_port_check(dev, port_num, flags)) {
                rsc = &resources[resource_index++];
                status = uct_ib_device_port_get_resource(dev, port_num, rsc);
                if (status != UCS_OK) {
                    goto err_free;
                }
            }
        }
    }

    *num_resources_p = num_resources;
    *resources_p     = resources;
    return UCS_OK;

err_free:
    ucs_free(resources);
err:
    return status;
}

static void uct_ib_register_tls(uct_context_t *context)
{
#if HAVE_TL_RC
    extern uct_tl_ops_t uct_rc_tl_ops;
    uct_register_tl(context, "rc", &uct_rc_tl_ops);
#endif
}

ucs_status_t uct_ib_init(uct_context_h context)
{
    uct_ib_context_t *ibctx = ucs_component_get(context, ib, uct_ib_context_t);
    struct ibv_device **device_list;
    ucs_status_t status;
    int i, num_devices;

    /* Get device list from driver */
    device_list = ibv_get_device_list(&num_devices);
    if (device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    /* Allocate array for devices */
    ibctx->devices = ucs_calloc(num_devices, sizeof(uct_ib_device_t*), "ib device list");
    if (ibctx->devices == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_device_list;
    }

    /* Initialize all devices */
    /* TODO apply a user-defined regex/wildcard filter */
    ibctx->num_devices = 0;
    for (i = 0; i < num_devices; ++i) {
        status = uct_ib_device_create(device_list[i],
                                      &ibctx->devices[ibctx->num_devices]);
        if (status != UCS_OK) {
            ucs_warn("Failed to initialize %s (%s), ignoring it",
                     ibv_get_device_name(device_list[i]),
                     ucs_status_string(status));
        } else {
            ++ibctx->num_devices;
        }
    }

    /* If we don't have any IB devices, fail the component */
    if (ibctx->num_devices > 0) {
        ucs_debug("initialized IB component with %u devices", ibctx->num_devices);
        uct_ib_register_tls(context);
        status = UCS_OK;
    } else {
        ucs_free(ibctx->devices);
        status = UCS_ERR_NO_DEVICE;
    }

out_free_device_list:
    ibv_free_device_list(device_list);
out:
    return status;
}

void uct_ib_cleanup(uct_context_t *context)
{
    uct_ib_context_t *ibctx = ucs_component_get(context, ib, uct_ib_context_t);
    unsigned i;

    for (i = 0; i < ibctx->num_devices; ++i) {
        uct_ib_device_destroy(ibctx->devices[i]);
    }
    ucs_free(ibctx->devices);
}

UCS_COMPONENT_DEFINE(uct_context_t, ib, uct_ib_init, uct_ib_cleanup, uct_ib_context_t)

