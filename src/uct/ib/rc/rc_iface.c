/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


ucs_status_t uct_rc_iface_open(uct_context_h context, const char *hw_name,
                               uct_iface_h *iface_p)
{
    ucs_status_t status;
    uct_rc_iface_t *iface;

    iface = ucs_malloc(sizeof(*iface), "rc iface");
    if (iface == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    iface->super.super.ops.iface_close         = uct_rc_iface_close;
    iface->super.super.ops.iface_get_address   = uct_rc_iface_get_address;
    iface->super.super.ops.iface_flush         = uct_rc_iface_flush;
    iface->super.super.ops.ep_get_address      = uct_rc_ep_get_address;
    iface->super.super.ops.ep_connect_to_iface = NULL;
    iface->super.super.ops.ep_connect_to_ep    = uct_rc_ep_connect_to_ep;

    status = ucs_ib_iface_init(context, &iface->super, hw_name);
    if (status != UCS_OK) {
        goto err_free;
    }

    ucs_debug("opened RC dev %s port %d",
              uct_ib_device_name(uct_ib_iface_device(&iface->super)),
              iface->super.port_num);

    *iface_p = &iface->super.super;
    return UCS_OK;

err_free:
    ucs_free(iface);
    return status;
}

void uct_rc_iface_close(uct_iface_h tl_iface)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    ucs_ib_iface_cleanup(&iface->super);
    ucs_free(iface);
}

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    iface_attr->max_short      = 0;
    iface_attr->max_bcopy      = 0;
    iface_attr->max_zcopy      = 0;
    iface_attr->iface_addr_len = sizeof(uct_ib_iface_addr_t);
    iface_attr->ep_addr_len    = sizeof(uct_rc_ep_addr_t);
    iface_attr->flags          = 0;
}

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    *(uct_ib_iface_addr_t*)iface_addr = iface->super.addr;
    return UCS_OK;
}

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface, uct_req_h *req_p,
                                uct_completion_cb_t cb)
{
    return UCS_OK;
}
