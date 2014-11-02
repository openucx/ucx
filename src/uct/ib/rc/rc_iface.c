/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_iface.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>

extern uct_tl_ops_t uct_rc_tl_ops;

ucs_status_t uct_rc_iface_open(uct_context_h context, const char *hw_name,
                               uct_iface_h *iface_p)
{
    ucs_status_t status;
    uct_rc_iface_t *iface;

    iface = ucs_malloc(sizeof(*iface), "rc iface");
    if (iface == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    iface->super.super.ops = &uct_rc_tl_ops;
    status = ucs_ib_iface_init(context, &iface->super, hw_name);
    if (status != UCS_OK) {
        goto err_free;
    }

    ucs_debug("opened RC dev %s port %d", uct_ib_device_name(iface->super.device),
              iface->super.port_num);

    *iface_p = &iface->super.super;
    return UCS_OK;

err_free:
    ucs_free(iface);
    return status;
}

void uct_rc_iface_close(uct_iface_h iface)
{
    ucs_free(iface);
}

