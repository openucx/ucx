/**
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_EFA_H_
#define UCT_IB_EFA_H_

#include <ucs/type/status.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_md.h>
#include <infiniband/efadv.h>


#define UCT_IB_AWS_EFA_PCI_VENDOR_ID 0x1d0f


typedef struct uct_ib_efadv_md {
    uct_ib_md_t super;
} uct_ib_efadv_md_t;


/* Devices with an unavailable PCI vendor id are not filtered out, and have to
 * be probed by efadv_query_device() */
static inline int
uct_ib_efadv_pci_vendor_match(const ucs_sys_pci_id_t *pci_id)
{
    return (pci_id->vendor == UCS_SYS_PCI_ID_VALUE_UNDEFINED) ||
           (pci_id->vendor == UCT_IB_AWS_EFA_PCI_VENDOR_ID);
}


static inline int
uct_ib_efadv_has_rdma_read(const struct efadv_device_attr *dev_attr)
{
#if HAVE_DECL_EFADV_DEVICE_ATTR_CAPS_RDMA_READ
    return !!(dev_attr->device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_READ);
#else
    return 0;
#endif
}


static inline int
uct_ib_efadv_has_rdma_write(const struct efadv_device_attr *dev_attr)
{
#if HAVE_DECL_EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE
    return !!(dev_attr->device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE);
#else
    return 0;
#endif
}


extern uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(efa);

#endif
