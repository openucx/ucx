/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef SRD_DEF_H_
#define SRD_DEF_H_

#include <uct/ib/base/ib_iface.h>
#include <ucs/arch/cpu.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/frag_list.h>
#include <ucs/sys/math.h>


#define UCT_SRD_INITIAL_PSN      1
#define UCT_SRD_RX_BATCH_MIN     8
#define UCT_SRD_SKB_ALIGN        UCS_SYS_CACHE_LINE_SIZE


typedef ucs_frag_list_sn_t        uct_srd_psn_t;
typedef struct uct_srd_iface      uct_srd_iface_t;
typedef struct uct_srd_ep         uct_srd_ep_t;
typedef struct uct_srd_ctl_hdr    uct_srd_ctl_hdr_t;
typedef struct uct_srd_iface_addr uct_srd_iface_addr_t;
typedef struct uct_srd_ep_addr    uct_srd_ep_addr_t;
typedef struct uct_srd_iface_peer uct_srd_iface_peer_t;

enum {
    UCT_SRD_PACKET_DEST_ID_SHIFT   = 24,
    UCT_SRD_PACKET_AM_ID_SHIFT     = 27,
};

enum {
    UCT_SRD_PACKET_FLAG_AM        = UCS_BIT(24),
    UCT_SRD_PACKET_FLAG_CTLX      = UCS_BIT(25),

    UCT_SRD_PACKET_AM_ID_MASK     = UCS_MASK(UCT_SRD_PACKET_AM_ID_SHIFT),
    UCT_SRD_PACKET_DEST_ID_MASK   = UCS_MASK(UCT_SRD_PACKET_DEST_ID_SHIFT),
};

enum {
    UCT_SRD_PACKET_CREQ = 1,
    UCT_SRD_PACKET_CREP = 2,
};

/*
network header layout

P - put emulation (will be disabled in the future)
C - control packet extended header

Active message packet header

 3         2 2 2 2             1 1
 1         6 5 4 3             6 5                             0
+---------------------------------------------------------------+
| am_id   |rsv|1|            dest_ep_id (24 bit)                |
+---------------------------------------------------------------+
|       psn (16 bit)        |
+----------------------------

Control packet header

 3   2 2 2 2 2 2 2             1 1
 1   9 8 7 6 5 4 3             6 5                             0
+---------------------------------------------------------------+
|rsv|C|0|                    dest_ep_id (24 bit)                |
+---------------------------------------------------------------+
|       psn (16 bit)        |
+----------------------------

    // neth layout in human readable form
    uint32_t           dest_ep_id:24;
    uint8_t            is_am:1;
    union {
        struct { // am false
            uint8_t ctl:1;
            uint8_t reserved:6;
        } ctl;
        struct { // am true
            uint8_t reserved:2;
            uint8_t am_id:5;
        } am;
    };
*/

typedef struct uct_srd_neth {
    uint32_t             packet_type;
    uct_srd_psn_t        psn;
} UCS_S_PACKED uct_srd_neth_t;


enum {
    UCT_SRD_SEND_SKB_FLAG_COMP       = UCS_BIT(0), /* This skb contains a completion */
    UCT_SRD_SEND_SKB_FLAG_ZCOPY      = UCS_BIT(1), /* This skb contains a zero-copy segment */
    UCT_SRD_SEND_SKB_FLAG_FLUSH      = UCS_BIT(2), /* This skb is a dummy flush skb */

#if UCS_ENABLE_ASSERT
    UCT_SRD_SEND_SKB_FLAG_INVALID    = UCS_BIT(7)  /* skb is released */
#else
    UCT_SRD_SEND_SKB_FLAG_INVALID    = 0
#endif
};


/*
 * Send skb with completion layout:
 * - if COMP skb flag is set, skb contains uct_srd_comp_desc_t after the payload
 * - if ZCOPY skb flag is set, skb contains uct_srd_zcopy_desc_t after the payload.
 * - otherwise, there is no additional data.
 */
typedef struct uct_srd_send_skb {
    ucs_queue_elem_t        out_queue;  /* in ep outstanding send queue */
    uint16_t                sn;         /* iface sequence number */
    uint32_t                lkey;
    uint16_t                len;        /* data size */
    uint16_t                flags;
    uct_srd_ep_t            *ep;        /* ep that sends skb */
    uct_srd_neth_t          neth[0];
} UCS_S_PACKED UCS_V_ALIGNED(UCT_SRD_SKB_ALIGN) uct_srd_send_skb_t;


/*
 * Call user completion handler
 */
typedef struct uct_srd_comp_desc {
    uct_completion_t        *comp;
    ucs_status_t            status; /* used in case of failure */
} uct_srd_comp_desc_t;


/**
 * Used to keep uct_iov_t buffers without datatype information.
 */
typedef struct uct_srd_iov {
    void                   *buffer; /* Data buffer */
    uint32_t               lkey;    /* Lkey for memory region */
    uint16_t               length;  /* Length of the buffer in bytes */
} UCS_S_PACKED uct_srd_iov_t;


typedef struct uct_srd_zcopy_desc {
    uct_srd_comp_desc_t      super;
    uct_srd_iov_t            iov[UCT_IB_MAX_IOV];
    uint16_t                 iovcnt; /* Count of the iov[] array valid elements */
} uct_srd_zcopy_desc_t;


typedef struct uct_srd_recv_skb {
    uct_ib_iface_recv_desc_t super;
    struct {
        ucs_frag_list_elem_t     elem;
    } ooo;
    struct {
        uint32_t                 len;
    } am;
} uct_srd_recv_skb_t;


typedef struct uct_srd_am_short_hdr {
    uint64_t hdr;
} UCS_S_PACKED uct_srd_am_short_hdr_t;


struct uct_srd_iface_addr {
    uct_ib_uint24_t     qp_num;
};


struct uct_srd_ep_addr {
    uct_srd_iface_addr_t iface_addr;
    uct_ib_uint24_t      ep_id;
};


static inline uint32_t uct_srd_neth_get_dest_id(uct_srd_neth_t *neth)
{
    return neth->packet_type & UCT_SRD_PACKET_DEST_ID_MASK;
}

static inline void uct_srd_neth_set_dest_id(uct_srd_neth_t *neth, uint32_t id)
{
    neth->packet_type |= id;
}

static inline uint8_t uct_srd_neth_get_am_id(uct_srd_neth_t *neth)
{
    return neth->packet_type >> UCT_SRD_PACKET_AM_ID_SHIFT;
}

static inline void uct_srd_neth_set_am_id(uct_srd_neth_t *neth, uint8_t id)
{
    neth->packet_type |= (id << UCT_SRD_PACKET_AM_ID_SHIFT);
}

static inline uct_srd_comp_desc_t *uct_srd_comp_desc(uct_srd_send_skb_t *skb)
{
    ucs_assert(skb->flags & UCT_SRD_SEND_SKB_FLAG_COMP);
    ucs_assert(!(skb->flags & UCT_SRD_SEND_SKB_FLAG_INVALID));
    return (uct_srd_comp_desc_t*)((char*)skb->neth + skb->len);
}

static inline uct_srd_zcopy_desc_t *uct_srd_zcopy_desc(uct_srd_send_skb_t *skb)
{
    ucs_assert(skb->flags & UCT_SRD_SEND_SKB_FLAG_ZCOPY);
    ucs_assert(!(skb->flags & UCT_SRD_SEND_SKB_FLAG_INVALID));
    return (uct_srd_zcopy_desc_t*)((char*)skb->neth + skb->len);
}

#endif
