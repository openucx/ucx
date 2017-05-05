/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_DEVICE_H
#define UCT_UGNI_DEVICE_H

#include "ugni_types.h"
#include <uct/api/uct.h>

ucs_status_t uct_ugni_device_create(int dev_id, int index, uct_ugni_device_t *dev_p);
void uct_ugni_device_destroy(uct_ugni_device_t *dev);
void uct_ugni_device_get_resource(const char *tl_name, uct_ugni_device_t *dev,
                                  uct_tl_resource_desc_t *resource);
ucs_status_t uct_ugni_iface_get_dev_address(uct_iface_t *tl_iface, uct_device_addr_t *addr);
ucs_status_t uct_ugni_create_cdm(uct_ugni_cdm_t *cdm, uct_ugni_device_t *device, ucs_thread_mode_t thread_mode);
ucs_status_t uct_ugni_destroy_cdm(uct_ugni_cdm_t *cdm);
uct_ugni_device_t *uct_ugni_device_by_name(const char *dev_name);
ucs_status_t uct_ugni_query_tl_resources(uct_md_h md, const char *tl_name,
                                         uct_tl_resource_desc_t **resource_p,
                                         unsigned *num_resources_p);
ucs_status_t init_device_list();
ucs_status_t uct_ugni_create_md_cdm(uct_ugni_cdm_t *cdm);
ucs_status_t uct_ugni_create_cq(gni_cq_handle_t *cq, unsigned cq_size, uct_ugni_cdm_t *cdm);
ucs_status_t uct_ugni_destroy_cq(gni_cq_handle_t cq, uct_ugni_cdm_t *cdm);

#endif
