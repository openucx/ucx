/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SYSV_IFACE_H
#define UCT_SYSV_IFACE_H

#include <uct/sm/base/sm_iface.h>
#include "sysv_ep.h"


#define UCT_SYSV_TL_NAME    "sysv"


typedef struct uct_sysv_iface {
    uct_sm_iface_t          super; 
} uct_sysv_iface_t;


typedef struct uct_sysv_iface_config {
    uct_sm_iface_config_t      super; 
} uct_sysv_iface_config_t;


typedef struct uct_sysv_lkey {
    int                     shmid;
    void                    *owner_ptr;
} uct_sysv_lkey_t;


typedef struct uct_sysv_rkey {
    int                     shmid;
    uintptr_t               owner_ptr;
} uct_sysv_rkey_t;


#endif
