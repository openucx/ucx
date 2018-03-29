/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UD_DEF_H_
#define UD_DEF_H_

#include <uct/ib/base/ib_iface.h>
#include <ucs/arch/cpu.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/frag_list.h>
#include <ucs/sys/math.h>


#define UCT_UD_QP_HASH_SIZE     256
#define UCT_UD_TX_MODERATION    64
#define UCT_UD_MIN_INLINE       48
#define UCT_UD_HASH_SIZE        997
#define UCT_UD_RX_BATCH_MIN     8

#define UCT_UD_INITIAL_PSN      1   /* initial packet serial number */
/* congestion avoidance settings. See ud_ep.h for details */
#define UCT_UD_CA_AI_VALUE      1   /* window += AI_VALUE */
#define UCT_UD_CA_MD_FACTOR     2   /* window = window/factor */
#define UCT_UD_CA_DUP_ACK_CNT   2   /* TODO: not implemented yet */
#define UCT_UD_RESENDS_PER_ACK  4   /* request per every N resends */

/* note that the ud tx window is [acked_psn+1, max_psn)
 * and max_psn = acked_psn + cwnd
 * so add 1 to the max/min window constants instead of doing this in the code
 */
#define UCT_UD_CA_MIN_WINDOW    2
#define UCT_UD_CA_MAX_WINDOW    1025


typedef uint16_t                 uct_ud_psn_t;
#define UCT_UD_PSN_COMPARE       UCS_CIRCULAR_COMPARE16
typedef struct uct_ud_iface      uct_ud_iface_t;
typedef struct uct_ud_ep         uct_ud_ep_t;
typedef struct uct_ud_ctl_hdr    uct_ud_ctl_hdr_t;
typedef uct_ib_qpnum_t           uct_ud_iface_addr_t;
typedef struct uct_ud_ep_addr    uct_ud_ep_addr_t;
typedef struct uct_ud_iface_peer uct_ud_iface_peer_t;

enum {
    UCT_UD_PACKET_ACK_REQ_SHIFT   = 25,
    UCT_UD_PACKET_AM_ID_SHIFT     = 27,
    UCT_UD_PACKET_DEST_ID_SHIFT   = 24,
    UCT_UD_PACKET_PUT_SHIFT       = 28,
};

enum {
    UCT_UD_PACKET_FLAG_AM      = UCS_BIT(24),
    UCT_UD_PACKET_FLAG_ACK_REQ = UCS_BIT(25),
    UCT_UD_PACKET_FLAG_ECN     = UCS_BIT(26),
    UCT_UD_PACKET_FLAG_NAK     = UCS_BIT(27),
    UCT_UD_PACKET_FLAG_PUT     = UCS_BIT(28),
    UCT_UD_PACKET_FLAG_CTL     = UCS_BIT(29),

    UCT_UD_PACKET_AM_ID_MASK     = UCS_MASK(UCT_UD_PACKET_AM_ID_SHIFT),
    UCT_UD_PACKET_DEST_ID_MASK   = UCS_MASK(UCT_UD_PACKET_DEST_ID_SHIFT),
};

enum {
    UCT_UD_PACKET_CREQ = 1,
    UCT_UD_PACKET_CREP = 2,
};

/*
network header layout

A - ack request
E - explicit congestion notification (ecn)
N - negative acknoledgement
P - put emulation (will be disabled in the future)
C - control packet extended header

Active message packet header

 3         2 2 2 2             1 1
 1         6 5 4 3             6 5                             0
+---------------------------------------------------------------+
| am_id   |E|A|1|            dest_ep_id (24 bit)                |
+---------------------------------------------------------------+
|       ack_psn (16 bit)        |           psn (16 bit)        |
+---------------------------------------------------------------+

Control packet header

 3   2 2 2 2 2 2 2             1 1
 1   9 8 7 6 5 4 3             6 5                             0
+---------------------------------------------------------------+
|rsv|C|P|N|E|A|0|            dest_ep_id (24 bit)                |
+---------------------------------------------------------------+
|       ack_psn (16 bit)        |           psn (16 bit)        |
+---------------------------------------------------------------+

    // neth layout in human readable form
    uint32_t           dest_ep_id:24;
    uint8_t            is_am:1;
    union {
        struct { // am false
            uint8_t ack_req:1;
            uint8_t ecn:1;
            uint8_t nak:1;
            uint8_t put:1;
            uint8_t ctl:1;
            uint8_t reserved:2;
        } ctl;
        struct { // am true
            uint8_t ack_req:1;
            uint8_t ecn:1;
            uint8_t am_id:5;
        } am;
    };
*/

typedef struct uct_ud_neth {
    uint32_t            packet_type;
    uct_ud_psn_t        psn;
    uct_ud_psn_t        ack_psn;
} UCS_S_PACKED uct_ud_neth_t;


enum {
    UCT_UD_SEND_SKB_FLAG_ACK_REQ = UCS_BIT(1), /* ACK was requested for this skb */
    UCT_UD_SEND_SKB_FLAG_COMP    = UCS_BIT(2), /* This skb contains a completion */
    UCT_UD_SEND_SKB_FLAG_ZCOPY   = UCS_BIT(3), /* This skb contains a zero-copy segment */
    UCT_UD_SEND_SKB_FLAG_ERR     = UCS_BIT(4), /* This skb contains a status after failure */
    UCT_UD_SEND_SKB_FLAG_CANCEL  = UCS_BIT(5)  /* This skb contains a UCS_ERR_CANCEL status */
};


/*
 * Send skb with completion layout:
 * - if COMP skb flag is set, skb contains uct_ud_comp_desc_t after the payload
 * - if ZCOPY skb flag is set, skb contains uct_ud_zcopy_desc_t after the payload.
 * - otherwise, there is no additional data.
 */
typedef struct uct_ud_send_skb {
    ucs_queue_elem_t        queue;      /* in send window */
    uint32_t                lkey;
    uint16_t                len;        /* data size */
    uint8_t                 flags;
    int8_t                  status;     /* used in case of failure */
    uct_ud_neth_t           neth[0];
} UCS_S_PACKED UCS_V_ALIGNED(UCS_SYS_CACHE_LINE_SIZE) uct_ud_send_skb_t;


typedef struct uct_ud_comp_desc {
    uct_completion_t        *comp;
    uct_ud_ep_t             *ep;
} uct_ud_comp_desc_t;


/**
 * Used to keep uct_iov_t buffers without datatype information.
 */
typedef struct uct_ud_iov {
    void                   *buffer;   /**< Data buffer */
    uint16_t                length;   /**< Length of the buffer in bytes */
} UCS_S_PACKED uct_ud_iov_t;


typedef struct uct_ud_zcopy_desc {
    uct_ud_comp_desc_t      super;
    uct_ud_iov_t            iov[UCT_IB_MAX_IOV];
    uint16_t                iovcnt; /* Count of the iov[] array valid elements */
} uct_ud_zcopy_desc_t;


typedef struct uct_ud_send_skb_inl {
    uct_ud_send_skb_t  super;
    char               data[sizeof(uct_ud_neth_t)]; /* placeholder for super.neth */
} uct_ud_send_skb_inl_t;


typedef struct uct_ud_recv_skb {
    uct_ib_iface_recv_desc_t super;
    union {
        struct {
            ucs_frag_list_elem_t     elem;
        } ooo;
        struct {
            ucs_queue_elem_t         queue;
            uint32_t                 len;
        } am;
    } u;
} uct_ud_recv_skb_t;


typedef struct uct_ud_am_short_hdr {
    uint64_t hdr;
} UCS_S_PACKED uct_ud_am_short_hdr_t;


typedef struct uct_ud_put_hdr {
    uint64_t rva;
} UCS_S_PACKED uct_ud_put_hdr_t;


struct uct_ud_iface_addr {
    uct_ib_uint24_t     qp_num;
};


struct uct_ud_ep_addr {
    uct_ud_iface_addr_t iface_addr;
    uct_ib_uint24_t     ep_id;
};


static inline uint32_t uct_ud_neth_get_dest_id(uct_ud_neth_t *neth)
{
    return neth->packet_type & UCT_UD_PACKET_DEST_ID_MASK;
}

static inline void uct_ud_neth_set_dest_id(uct_ud_neth_t *neth, uint32_t id)
{
    neth->packet_type |= id;
}

static inline uint8_t uct_ud_neth_get_am_id(uct_ud_neth_t *neth)
{
    return neth->packet_type >> UCT_UD_PACKET_AM_ID_SHIFT;
}

static inline void uct_ud_neth_set_am_id(uct_ud_neth_t *neth, uint8_t id)
{
    neth->packet_type |= (id << UCT_UD_PACKET_AM_ID_SHIFT);
}

static inline uct_ud_comp_desc_t *uct_ud_comp_desc(uct_ud_send_skb_t *skb)
{
    ucs_assert(skb->flags & (UCT_UD_SEND_SKB_FLAG_COMP  |
                             UCT_UD_SEND_SKB_FLAG_ERR   |
                             UCT_UD_SEND_SKB_FLAG_CANCEL));
    return (uct_ud_comp_desc_t*)((char *)skb->neth + skb->len);
}

static inline uct_ud_zcopy_desc_t *uct_ud_zcopy_desc(uct_ud_send_skb_t *skb)
{
    ucs_assert(skb->flags & UCT_UD_SEND_SKB_FLAG_ZCOPY);
    return (uct_ud_zcopy_desc_t*)((char *)skb->neth + skb->len);
}


#endif
