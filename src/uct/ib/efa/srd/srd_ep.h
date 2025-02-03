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

enum {
    UCT_SRD_EP_FLAG_CONNECTED   = UCS_BIT(0), /* EP is connected to the peer */
    UCT_SRD_EP_FLAG_PRIVATE     = UCS_BIT(1), /* EP was created as internal */
    UCT_SRD_EP_FLAG_HAS_PENDING = UCS_BIT(2), /* EP has some pending requests */
};

typedef uint32_t uct_srd_ep_conn_sn_t;

typedef struct uct_srd_ep {
    uct_base_ep_t             super;
    uct_srd_ep_conn_sn_t      conn_sn;

    uint32_t                  ep_id;
    uint32_t                  dest_ep_id;
    uint8_t                   path_index;

    /* connection sequence number. assigned in connect_to_iface() */
    uint8_t                   rx_creq_count;
    uint16_t                  flags;
    uct_srd_ep_peer_address_t peer_address;

    struct {
        ucs_frag_list_t       ooo_pkts; /* Out of order packets that
                                           can not be processed yet */
    } rx;
    struct {
        uct_srd_psn_t         psn;      /* Next PSN to send */
    } tx;
} uct_srd_ep_t;


ucs_status_t uct_srd_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr);
ucs_status_t
uct_srd_ep_connect_to_ep_v2(uct_ep_h tl_ep, const uct_device_addr_t *dev_addr,
                            const uct_ep_addr_t *uct_ep_addr,
                            const uct_ep_connect_to_ep_params_t *params);

UCS_CLASS_DECLARE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);

#endif
