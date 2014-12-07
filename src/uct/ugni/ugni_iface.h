/**
 * Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_UGNI_IFACE_H
#define UCT_UGNI_IFACE_H

#include "ugni_device.h"

typedef struct uct_ugni_iface_addr {
    uct_iface_addr_t    super;
    /* TBD */
} uct_ugni_iface_addr_t;

typedef struct uct_ugni_iface {
    uct_iface_t             super;
    uct_ugni_device_t       *dev;
    uct_ugni_iface_addr_t   addr;
    /* TBD */
    /* list of ep */
} uct_ugni_iface_t;

typedef struct uct_ugni_iface_config {
    uct_iface_config_t      super;
} uct_ugni_iface_config_t;

static inline uct_ugni_device_t * uct_ugni_iface_device(uct_ugni_iface_t *iface)
{
    return iface->dev;
}

extern ucs_config_field_t uct_ugni_iface_config_table[];
extern uct_tl_ops_t uct_ugni_tl_ops;

#if 0
void uct_ugni_iface_query(uct_ugni_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

uct_ugni_ep_t *uct_ugni_iface_lookup_ep(uct_ugni_iface_t *iface, unsigned qp_num);

void uct_ugni_iface_add_ep(uct_ugni_iface_t *iface, uct_ugni_ep_t *ep);
void uct_ugni_iface_remove_ep(uct_ugni_iface_t *iface, uct_ugni_ep_t *ep);
#endif

#endif
