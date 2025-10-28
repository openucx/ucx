/*
 * Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GAUDI_GDR_IFACE_H
#define UCT_GAUDI_GDR_IFACE_H

#include <uct/base/uct_iface.h>

#define UCT_GAUDI_GDR_TL_NAME    "gaudi_gdr"

typedef struct uct_gaudi_gdr_iface {
    uct_base_iface_t            super;
} uct_gaudi_gdr_iface_t;

#endif
