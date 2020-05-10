/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_KNEM_IFACE_H
#define UCT_KNEM_IFACE_H

#include "knem_md.h"

#include <uct/base/uct_iface.h>
#include <uct/sm/scopy/base/scopy_iface.h>


typedef struct uct_knem_iface_config {
    uct_scopy_iface_config_t      super;
} uct_knem_iface_config_t;


typedef struct uct_knem_iface {
    uct_scopy_iface_t             super;
    uct_knem_md_t                 *knem_md;
} uct_knem_iface_t;


#endif
