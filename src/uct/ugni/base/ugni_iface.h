/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_IFACE_H
#define UCT_UGNI_IFACE_H

#include "ugni_def.h"
#include "ugni_types.h"
#include <uct/base/uct_iface.h>
#include <uct/api/uct.h>

UCS_CLASS_DECLARE(uct_ugni_iface_t, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, uct_iface_ops_t*,
                  const uct_iface_config_t* UCS_STATS_ARG(ucs_stats_node_t*))

ucs_status_t uct_ugni_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp);
ucs_status_t uct_ugni_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp);
ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr);
int uct_ugni_iface_is_reachable(uct_iface_h tl_iface, const uct_device_addr_t *dev_addr, 
				const uct_iface_addr_t *iface_addr);
void uct_ugni_progress(void *arg);
ucs_status_t uct_ugni_fetch_pmi();
void uct_ugni_base_desc_init(ucs_mpool_t *mp, void *obj, void *chunk);
void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh);
ucs_status_t uct_ugni_query_tl_resources(uct_md_h md, const char *tl_name,
                                         uct_tl_resource_desc_t **resource_p,
                                         unsigned *num_resources_p);
static inline uct_ugni_device_t *uct_ugni_iface_device(uct_ugni_iface_t *iface)
{
    return iface->cdm.dev;
}
static inline gni_nic_handle_t uct_ugni_iface_nic_handle(uct_ugni_iface_t *iface)
{
    return iface->cdm.nic_handle;
}
static inline int uct_ugni_check_device_type(uct_ugni_iface_t *iface, gni_nic_device_t type)
{
    uct_ugni_device_t *dev = uct_ugni_iface_device(iface);
    return dev->type == type;
}

#endif
