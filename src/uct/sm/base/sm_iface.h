/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SM_IFACE_H
#define UCT_SM_IFACE_H

#include <uct/tl/tl_base.h>

struct uct_sm_iface;

typedef struct uct_sm_iface_addr {
    uct_iface_addr_t        super;
    uint32_t                nic_addr;
} uct_sm_iface_addr_t;

typedef struct uct_sm_pd {
    uct_pd_t                super;
} uct_sm_pd_t;

typedef struct uct_sm_iface {
    uct_base_iface_t        super;
    uct_sm_pd_t             pd;
    uct_sm_iface_addr_t     addr;
    struct {
        unsigned            max_put;
        unsigned            max_bcopy;
        unsigned            max_zcopy;
    } config;
} uct_sm_iface_t;

typedef struct uct_sm_iface_config {
    uct_iface_config_t      super;
} uct_sm_iface_config_t;

/* to make it visible to derived classes */
UCS_CLASS_DECLARE(uct_sm_iface_t, uct_iface_ops_t *, uct_worker_h, uct_pd_h, 
                  const uct_iface_config_t *, const char *, const char *)

ucs_status_t uct_sm_iface_query(uct_iface_h, uct_iface_attr_t *);
ucs_status_t uct_sm_iface_flush(uct_iface_h);
ucs_status_t uct_sm_iface_get_address(uct_iface_h, uct_iface_addr_t *);

#endif
