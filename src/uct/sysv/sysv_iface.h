/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
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


struct uct_sysv_iface;

typedef struct uct_sysv_iface_addr {
    uct_iface_addr_t        super;
    uint32_t                nic_addr;
} uct_sysv_iface_addr_t;

typedef struct uct_sysv_pd {
    uct_pd_t                super;
} uct_sysv_pd_t;

typedef struct uct_sysv_iface {
    uct_base_iface_t        super;
    uct_sysv_pd_t           pd;
    uct_sysv_iface_addr_t   addr;
    struct {
        unsigned            max_put;
        unsigned            max_bcopy;
        unsigned            max_zcopy;
    } config;
} uct_sysv_iface_t;

typedef struct uct_sysv_iface_config {
    uct_iface_config_t      super;
} uct_sysv_iface_config_t;

typedef struct uct_sysv_lkey {
    int                     shmid;
    void                    *owner_ptr;
} uct_sysv_lkey_t;

typedef struct uct_sysv_rkey {
    long long int           magic;
    int                     shmid;
    uintptr_t               owner_ptr;
} uct_sysv_rkey_t;

extern ucs_config_field_t uct_sysv_iface_config_table[];
extern uct_tl_ops_t uct_sysv_tl_ops;

#endif
