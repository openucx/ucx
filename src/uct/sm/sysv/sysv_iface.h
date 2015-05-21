/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SYSV_IFACE_H
#define UCT_SYSV_IFACE_H

#include <uct/sm/base/sm_iface.h>
#include "sysv_context.h"
#include "sysv_ep.h"


struct uct_sysv_iface;

typedef struct uct_sysv_pd {
    uct_pd_t                super;
} uct_sysv_pd_t;

typedef struct uct_sysv_iface {
    uct_sm_iface_t          super; 
    uct_sysv_pd_t           pd;
} uct_sysv_iface_t;

typedef struct uct_sysv_iface_config {
    uct_sm_iface_config_t      super; 
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

/* FIXME would like to point config table to base class. is it even needed? */
//extern ucs_config_field_t uct_sysv_iface_config_table[]; 
extern uct_tl_ops_t uct_sysv_tl_ops;

#endif
