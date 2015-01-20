/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_MMP_IFACE_H
#define UCT_MMP_IFACE_H

#include <uct/tl/tl_base.h>
#include "mmp_context.h"
#include "mmp_device.h"
#include "mmp_ep.h"

#define UCT_mmp_HASH_SIZE   256

struct uct_mmp_iface;

typedef struct uct_mmp_iface_addr {
    uct_iface_addr_t    super;
    uint32_t            nic_addr;
} uct_mmp_iface_addr_t;

typedef struct uct_mmp_pd {
    uct_pd_t      super;
    struct uct_mmp_iface *iface;
} uct_mmp_pd_t;

typedef struct uct_mmp_iface {
    uct_base_iface_t       super;
    uct_mmp_device_t       *dev;
    uct_mmp_pd_t           pd;
    uct_mmp_iface_addr_t   addr;
    gni_cdm_handle_t       cdm_handle;      /**< mmp communication domain */
    gni_nic_handle_t       nic_handle;      /**< mmp NIC handle */
    uint32_t               pe_address;      /**< PE address for the NIC that
                                              this function has attached to the
                                              communication domain. */
    uct_mmp_iface_addr_t   address;         /**< PE address that is returned
                                              for the communication domain that
                                              this NIC is attached to. */
    gni_cq_handle_t        local_cq;        /**< Completion queue */
    int                    domain_id;       /**< Id for mmp domain creation */
    uct_mmp_ep_t           *eps[UCT_mmp_HASH_SIZE];    /**< Array of QPs */
    unsigned               outstanding;     /**< Counter for outstanding
                                              packets on the interface */
    ucs_mpool_h            free_fma_out;    /**< Pool of FMA descriptors 
                                                for outbound */
    bool                   activated;       /**< nic status */
    /* list of ep */
} uct_mmp_iface_t;

typedef struct uct_mmp_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mpool;
} uct_mmp_iface_config_t;

typedef struct uct_mmp_fma_desc {
    gni_post_descriptor_t desc;
    uct_mmp_ep_t  *ep;
} uct_mmp_fma_desc_t;

static inline uct_mmp_device_t * uct_mmp_iface_device(uct_mmp_iface_t *iface)
{
    return iface->dev;
}

extern ucs_config_field_t uct_mmp_iface_config_table[];
extern uct_tl_ops_t uct_mmp_tl_ops;

ucs_status_t uct_mmp_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                 uct_rkey_bundle_t *rkey_ob);

ucs_status_t mmp_activate_iface(uct_mmp_iface_t *iface, uct_mmp_context_t
                                *mmp_ctx);
#endif
