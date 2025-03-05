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


typedef struct uct_srd_hdr {
    uint64_t        ep_uuid; /* Sender EP's random identifier */
    uct_srd_psn_t   psn;     /* Sender EP's packet sequence number */
    uint8_t         id;      /* AM and flags */
} UCS_S_PACKED uct_srd_hdr_t;


typedef struct uct_srd_am_short_hdr {
    uct_srd_hdr_t   srd_hdr;
    uint64_t        am_hdr;
} UCS_S_PACKED uct_srd_am_short_hdr_t;


typedef struct uct_srd_iface_addr {
    uct_ib_uint24_t qp_num;
} uct_srd_iface_addr_t;


#endif
