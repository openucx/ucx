/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_CMA_EP_H
#define UCT_CMA_EP_H

#include "cma_iface.h"

#include <uct/tl/tl_log.h>


typedef struct uct_cma_ep {
    uct_base_ep_t super;
    pid_t         remote_pid;
} uct_cma_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_cma_ep_t, uct_ep_t, uct_iface_t*,
                           const struct sockaddr *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_cma_ep_t, uct_ep_t);
ucs_status_t uct_cma_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_cma_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp);
#endif
