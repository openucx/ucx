/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_MM_IFACE_H
#define UCT_MM_IFACE_H

#include <uct/tl/tl_base.h>

#define UCT_MM_TL_NAME "mm"


typedef struct uct_mm_iface_config {
    uct_iface_config_t      super;
} uct_mm_iface_config_t;


typedef struct uct_mm_iface {
    uct_base_iface_t        super;
    /* TODO shared FIFO id and address */
} uct_mm_iface_t;


extern uct_tl_component_t uct_mm_tl;

#endif
