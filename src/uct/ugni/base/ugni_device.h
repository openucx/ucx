/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_DEVICE_H
#define UCT_UGNI_DEVICE_H

#include <stdbool.h>
#include <gni_pub.h>

#include "uct/api/uct.h"


#define UCT_UGNI_MAX_TYPE_NAME     (10)
#define UCT_UGNI_MD_NAME   "ugni"

typedef struct uct_ugni_device {
    gni_nic_device_t type;                      /**< Device type */
    char             type_name[UCT_UGNI_MAX_TYPE_NAME];  /**< Device type name */
    char             fname[UCT_DEVICE_NAME_MAX];/**< Device full name */
    uint32_t         device_id;                 /**< Device id */
    int              device_index;              /**< Index of the device in the
                                                  array of devices */
    uint32_t         address;                   /**< Device address */
    uint32_t         cpu_id;                    /**< CPU attached directly
                                                  to the device */
    cpu_set_t        cpu_mask;                  /**< CPU mask */
    bool             attached;                  /**< device was attached */
    /* TBD - reference counter */
} uct_ugni_device_t;

ucs_status_t uct_ugni_device_create(int dev_id, int index, uct_ugni_device_t *dev_p);

void uct_ugni_device_destroy(uct_ugni_device_t *dev);

void uct_ugni_device_get_resource(const char *tl_name, uct_ugni_device_t *dev,
                                  uct_tl_resource_desc_t *resource);
#endif
