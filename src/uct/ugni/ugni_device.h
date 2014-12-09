/**
 * Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_UGNI_DEVICE_H
#define UCT_UGNI_DEVICE_H

#include <stdbool.h>
#include <gni_pub.h>

#include "uct/api/uct.h"


#define MAX_TYPE_NAME     (10)
#define TL_NAME           "ugni"

typedef struct uct_ugni_device {
    uct_pd_t         super;                     /**< Protection domain */
    gni_nic_device_t type;                      /**< Device type */
    char             type_name[MAX_TYPE_NAME];  /**< Device type name */
    char             fname[UCT_MAX_NAME_LEN];   /**< Device full name */
    uint32_t         device_id;                 /**< Device id */
    uint32_t         address;                   /**< Device address */
    uint32_t         cpu_id;                    /**< CPU attached directly
                                                  to the device */
    cpu_set_t        cpu_mask;                  /**< CPU mask */
    bool             attached;                  /**< device was attached */
    /* TBD - reference counter */
} uct_ugni_device_t;

ucs_status_t uct_ugni_device_create(int dev_id, uct_ugni_device_t *dev_p);

void uct_ugni_device_destroy(uct_ugni_device_t *dev);

void uct_device_get_resource(uct_ugni_device_t *dev,
                             uct_resource_desc_t *resource);

ucs_status_t uct_ugni_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                  uct_rkey_bundle_t *rkey_ob);
#endif
