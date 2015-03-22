/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_int.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <string.h>


ucs_status_t ucp_ep_create(ucp_worker_h worker, ucp_ep_h *ep_p)
{
    uct_iface_attr_t iface_attr;
    ucs_status_t status;
    ucp_ep_t *ep;

    ep = ucs_malloc(sizeof(*ep), "ucp ep");
    if (ep == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = uct_iface_query(worker->ifaces[0], &iface_attr);
    if (status != UCS_OK) {
        goto err_free_ep;
    }

    ep->worker               = worker;
    ep->config.max_short_tag = iface_attr.cap.am.max_short - sizeof(uint64_t);

    status = uct_ep_create(worker->ifaces[0], &ep->uct);
    if (status != UCS_OK) {
        goto err_free_ep;
    }

    *ep_p = ep;
    return UCS_OK;

err_free_ep:
    ucs_free(ep);
err:
    return status;
}

void ucp_ep_destroy(ucp_ep_h ep)
{
    uct_ep_destroy(ep->uct);
    ucs_free(ep);
}

size_t ucp_ep_address_length(ucp_ep_h ep)
{
    uct_iface_attr_t iface_attr;
    uct_iface_query(ep->uct->iface, &iface_attr);
    return iface_attr.iface_addr_len + iface_attr.ep_addr_len;
}

ucs_status_t ucp_ep_pack_address(ucp_ep_h ep, ucp_address_t *address)
{
    uct_iface_attr_t iface_attr;
    uct_iface_query(ep->uct->iface, &iface_attr);
    uct_iface_get_address(ep->uct->iface, (void*)address);
    return uct_ep_get_address(ep->uct, (void*)address + iface_attr.iface_addr_len);
}

ucs_status_t ucp_ep_connect(ucp_ep_h ep, ucp_address_t *dest_addr)
{
    uct_iface_attr_t iface_attr;
    uct_iface_query(ep->uct->iface, &iface_attr);
    return uct_ep_connect_to_ep(ep->uct, (void*)dest_addr,
                                (void*)dest_addr + iface_attr.iface_addr_len);
}
