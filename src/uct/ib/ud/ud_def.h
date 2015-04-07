/** 
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*   
* $COPYRIGHT$
* $HEADER$
*/
    
#ifndef UD_DEF_H_
#define UD_DEF_H_


#include <ucs/sys/math.h>

#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/frag_list.h>
#include <uct/ib/base/ib_iface.h>

#define UCT_UD_QP_HASH_SIZE     256    
#define UCT_UD_QKEY             0x1ee7a330
#define UCT_UD_MAX_SGE          2
#define UCT_UD_MAX_WINDOW       1024 /* TODO: make it config param */
#define UCT_UD_TX_MODERATION    64
#define UCT_UD_MIN_INLINE       48
#define UCT_UD_HASH_SIZE        251

typedef uint16_t                 uct_ud_psn_t;
#define UCT_UD_PSN_COMPARE       UCS_CIRCULAR_COMPARE16
typedef struct uct_ud_iface      uct_ud_iface_t;
typedef struct uct_ud_ep         uct_ud_ep_t;
typedef struct uct_ud_ctl_hdr    uct_ud_ctl_hdr_t;
typedef struct uct_ud_iface_addr uct_ud_iface_addr_t;
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

typedef struct uct_ud_send_skb {
    ucs_queue_elem_t        queue;      /* in send window */
    uint32_t                lkey;
    uint32_t                len;        /* data size */
    uct_ud_neth_t           neth[0];
} UCS_S_PACKED uct_ud_send_skb_t;

typedef struct uct_ud_recv_skb {
    uct_ib_iface_recv_desc_t super;
    ucs_frag_list_elem_t     ooo_elem;
} uct_ud_recv_skb_t;

typedef struct uct_ud_am_short_hdr {
    uint64_t hdr;
} UCS_S_PACKED uct_ud_am_short_hdr_t;

typedef struct uct_ud_put_hdr {
    uint64_t rva;
} UCS_S_PACKED uct_ud_put_hdr_t;



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

/* 
 * Allow sceduling of 'operation' on the interface. The operations
 * are executed in time of progress in round robin fashion. 
 */
enum {
    UCT_UD_EP_OP_NONE       = 0,
    UCT_UD_EP_OP_ACK        = UCS_BIT(0),  /* ack data */
    UCT_UD_EP_OP_ACK_REQ    = UCS_BIT(1),  /* request ack of sent packets */
    UCT_UD_EP_OP_RESEND     = UCS_BIT(2),  /* resend un acked packets */
    UCT_UD_EP_OP_CREP       = UCS_BIT(3),  /* send connection reply */
    UCT_UD_EP_OP_INPROGRESS = UCS_BIT(7)   /* bit is set when op is sceduled */
};

typedef struct uct_ud_ep_pending_op {
    ucs_queue_elem_t   queue;      
    uint32_t           ops;     /* bitmask that describes which op are sceduled */
} uct_ud_ep_pending_op_t;

#endif

