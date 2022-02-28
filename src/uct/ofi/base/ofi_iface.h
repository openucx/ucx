#ifndef UCT_OFI_IFACE_H
#define UCT_OFI_IFACE_H

#include "ofi_def.h"
#include "ofi_types.h"
#include "ofi_md.h"
#include <uct/base/uct_iface.h>
#include <uct/api/uct.h>

/*
UCS_CLASS_DECLARE(uct_ofi_iface_t, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, uct_iface_ops_t*,
                  const uct_iface_config_t*)
    */

ucs_status_t uct_ofi_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp);
ucs_status_t uct_ofi_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr);
int uct_ofi_iface_is_reachable(uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                                const uct_iface_addr_t *iface_addr);
void uct_ofi_base_desc_init(ucs_mpool_t *mp, void *obj, void *chunk);
void uct_ofi_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh);
void uct_ofi_cleanup_base_iface(uct_ofi_iface_t *iface);
int uct_ofi_get_next_av(uct_ofi_av_t *av);

#endif
