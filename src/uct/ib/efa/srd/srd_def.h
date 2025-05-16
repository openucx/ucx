/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SRD_DEF_H_
#define SRD_DEF_H_

#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/frag_list.h>


#define UCT_SRD_INITIAL_PSN     1
#define UCT_SRD_SEND_OP_ALIGN   UCS_SYS_CACHE_LINE_SIZE
#define UCT_SRD_SEND_DESC_ALIGN UCS_SYS_CACHE_LINE_SIZE


typedef ucs_frag_list_sn_t uct_srd_psn_t;

typedef struct uct_srd_ep        uct_srd_ep_t;
typedef struct uct_srd_send_op   uct_srd_send_op_t;
typedef struct uct_srd_send_desc uct_srd_send_desc_t;


typedef struct uct_srd_hdr {
    uint8_t         id;      /* AM and flags */
    uint64_t        ep_uuid; /* Sender EP's random identifier */
    uct_srd_psn_t   psn;     /* Sender EP's packet sequence number */
} UCS_S_PACKED uct_srd_hdr_t;


typedef enum uct_srd_ctl_id {
    UCT_SRD_CTL_ID_REQ = UCT_AM_ID_MAX,
    UCT_SRD_CTL_ID_RESP
} uct_srd_ctl_id_t;


typedef struct uct_srd_ctl_hdr {
    uint8_t         id;      /* Shared with @ref uct_srd_hdr_t::id */
    uct_ib_uint24_t qpn;     /* Sender qpn */
    uint64_t        ep_uuid;

    /* packed device address follows */
} UCS_S_PACKED uct_srd_ctl_hdr_t;


typedef struct uct_srd_ctl_op {
    ucs_queue_elem_t queue; /* Entry in iface tx pending control queue */
    struct ibv_ah    *ah;
    uint32_t         remote_qpn;
} uct_srd_ctl_op_t;


typedef struct uct_srd_am_short_hdr {
    uct_srd_hdr_t   srd_hdr;
    uint64_t        am_hdr;
} UCS_S_PACKED uct_srd_am_short_hdr_t;


typedef struct uct_srd_iface_addr {
    uct_ib_uint24_t qp_num;
} uct_srd_iface_addr_t;


typedef void (*uct_srd_send_op_comp_t)(uct_srd_send_op_t *send_op,
                                       ucs_status_t status);


/*
 * Send descriptor used when receiving TX CQE.
 */
struct uct_srd_send_op {
    ucs_list_link_t        list;       /* Link in ep outstanding send list */
    uct_srd_ep_t           *ep;        /* Sender EP */
    uct_completion_t       *user_comp; /* User completion, NULL if none */
    uct_srd_send_op_comp_t comp_cb;    /* Send operation completion */
} UCS_V_ALIGNED(UCT_SRD_SEND_OP_ALIGN);


/*
 * Registered send descriptor used for bcopy/zcopy posts and corresponding TX CQE.
 */
struct uct_srd_send_desc {
    uct_srd_send_op_t     super;
    uint32_t              lkey;        /* Registration key for this send_desc */
    uct_unpack_callback_t unpack_cb;
    void                  *unpack_arg;
    size_t                length;
} UCS_V_ALIGNED(UCT_SRD_SEND_DESC_ALIGN);


#endif
