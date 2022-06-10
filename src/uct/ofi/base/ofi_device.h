/**
 * Copyright (C) UT-Battelle, LLC. 2022. ALL RIGHTS
 */

#ifndef UCT_OFI_DEV_H
#define UCT_OFI_DEV_H
#include <ucs/type/status.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include "ofi_types.h"

ucs_status_t uct_ofi_init_fabric(uct_ofi_md_t*, char*);
ucs_status_t uct_ofi_query_devices(uct_md_h tl_md,
                                   uct_tl_device_resource_t **tl_devices_p,
                                   unsigned *num_tl_devices_p);

ucs_status_t uct_ofi_iface_get_dev_address(uct_iface_t *tl_iface, uct_device_addr_t *addr);
#endif
