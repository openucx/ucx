/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_IFACE_H
#define UCT_UGNI_IFACE_H

#include "ugni_types.h"
#include <uct/base/uct_iface.h>
#include <uct/api/uct.h>

UCS_CLASS_DECLARE(uct_ugni_iface_t, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, uct_iface_ops_t*,
                  const uct_iface_config_t* UCS_STATS_ARG(ucs_stats_node_t*))

ucs_status_t uct_ugni_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp);
ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr);
int uct_ugni_iface_is_reachable(uct_iface_h tl_iface, const uct_device_addr_t *dev_addr, 
				const uct_iface_addr_t *iface_addr);
void uct_ugni_base_desc_init(ucs_mpool_t *mp, void *obj, void *chunk);
void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh);
void uct_ugni_cleanup_base_iface(uct_ugni_iface_t *iface);
#define uct_ugni_iface_device(_iface) ((_iface)->cdm.dev)
#define uct_ugni_iface_nic_handle(_iface) ((_iface)->cdm.nic_handle)
#define uct_ugni_check_device_type(_iface, _type) ((_iface)->cdm.dev->type == _type)
#endif
