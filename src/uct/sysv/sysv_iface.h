/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SYSV_IFACE_H
#define UCT_SYSV_IFACE_H

#include <uct/tl/tl_base.h>
#include <ucs/sys/sys.h>
#include <stdbool.h>
#include "sysv_context.h"
#include "sysv_ep.h"


#define UCT_SYSV_HASH_SIZE   256

struct uct_sysv_iface;

typedef struct uct_sysv_iface_addr {
    uct_iface_addr_t    super;
    uint32_t            nic_addr;
} uct_sysv_iface_addr_t;

typedef struct uct_sysv_pd {
    uct_pd_t      super;
    struct uct_sysv_iface *iface;
} uct_sysv_pd_t;

typedef struct uct_sysv_iface {
    uct_base_iface_t       super;
    uct_sysv_pd_t           pd;
    uct_sysv_context_t      *ctx;
    uct_sysv_iface_addr_t   addr;
    uct_sysv_ep_t           *eps[UCT_SYSV_HASH_SIZE];    /**< Array of EPs */
    bool                   activated;                   /**< nic status */
    struct {
        unsigned            max_put;
    } config;
    /* list of ep */
} uct_sysv_iface_t;

typedef struct uct_sysv_iface_config {
    uct_iface_config_t       super;
} uct_sysv_iface_config_t;

extern ucs_config_field_t uct_sysv_iface_config_table[];
extern uct_tl_ops_t uct_sysv_tl_ops;

ucs_status_t uct_sysv_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                 uct_rkey_bundle_t *rkey_ob);

ucs_status_t sysv_activate_iface(uct_sysv_iface_t *iface,
                                uct_sysv_context_t *sysv_ctx);
#endif
