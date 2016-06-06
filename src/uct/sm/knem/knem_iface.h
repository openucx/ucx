/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_KNEM_IFACE_H
#define UCT_KNEM_IFACE_H

#include "knem_md.h"

#include <uct/base/uct_iface.h>


#define UCT_KNEM_TL_NAME "knem"


typedef struct uct_knem_iface_config {
    uct_iface_config_t      super;
} uct_knem_iface_config_t;


typedef struct uct_knem_iface {
    uct_base_iface_t        super;
    uct_knem_md_t        *knem_md;
} uct_knem_iface_t;


extern uct_tl_component_t uct_knem_tl;

#endif
