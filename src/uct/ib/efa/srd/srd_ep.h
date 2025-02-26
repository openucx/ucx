/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_EP_H
#define UCT_SRD_EP_H

#include "srd_def.h"


#define UCT_SRD_INITIAL_PSN 1
#define UCT_SRD_EP_NULL_ID  UCS_MASK(24)


typedef struct uct_srd_ep {
    uct_base_ep_t             super;
    uint64_t                  ep_uuid;        /* Random EP identifier */
    uint32_t                  ep_id;          /* Local interface EP index */
    uint8_t                   path_index;
    uct_srd_ep_peer_address_t peer_address;   /* Remote IFACE informations */

    struct {
        uct_srd_psn_t         psn;            /* Next PSN to send */
    } tx;
} uct_srd_ep_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);

#endif
