/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SYSV_DEVICE_H
#define UCT_SYSV_DEVICE_H

#include <stdbool.h>

#include "uct/api/uct.h"


#define MAX_TYPE_NAME     (10)
#define TL_NAME           "sysv"

typedef struct uct_sysv_device {
    uct_pd_t         super;                     /**< Protection domain */
    char             type_name[MAX_TYPE_NAME];  /**< Device type name */
    char             fname[UCT_MAX_NAME_LEN];   /**< Device full name */
} uct_sysv_device_t;

void uct_sysv_device_create(uct_context_h context, uct_sysv_device_t *dev_p);

void uct_sysv_device_destroy(uct_sysv_device_t *dev);

void uct_device_get_resource(uct_sysv_device_t *dev,
                             uct_resource_desc_t *resource);
#endif
