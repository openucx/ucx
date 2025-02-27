/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SRD_DEF_H_
#define SRD_DEF_H_

#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/frag_list.h>


typedef ucs_frag_list_sn_t uct_srd_psn_t;


enum {
    UCT_SRD_PACKET_FLAG_AM = UCS_BIT(UCT_AM_ID_BITS),
};


typedef struct uct_srd_neth {
    uint64_t        ep_uuid; /* Sender EP's random identifier */
    uct_srd_psn_t   psn;     /* Sender EP's packet sequence number */
    uint8_t         id;      /* AM and flags */
} UCS_S_PACKED uct_srd_neth_t;


typedef struct uct_srd_am_short_hdr {
    uct_srd_neth_t  neth;
    uint64_t        hdr;
} UCS_S_PACKED uct_srd_am_short_hdr_t;


typedef struct uct_srd_iface_addr {
    uct_ib_uint24_t qp_num;
} uct_srd_iface_addr_t;


typedef struct uct_srd_ep_peer_address {
    struct ibv_ah   *ah;
    uint32_t        dest_qpn;
} uct_srd_ep_peer_address_t;


#endif
