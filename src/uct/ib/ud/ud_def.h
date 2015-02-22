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
#include <ucs/datastruct/callbackq.h>
#include <uct/ib/base/ib_iface.h>

#define UCT_UD_QP_HASH_SIZE     256    
#define UCT_UD_QKEY             0x1ee7a330
#define UCT_UD_MAX_SGE          2
#define UCT_UD_MAX_WINDOW       1024 /* TODO: make it config param */
#define UCT_UD_TX_MODERATION    64
    
typedef uint16_t                uct_ud_psn_t;
#define UCT_UD_PSN_COMPARE      UCS_CIRCULAR_COMPARE16
typedef struct uct_ud_iface     uct_ud_iface_t;

enum {
    UCT_UD_PACKET_AM_ID_SHIFT   = 3,
    UCT_UD_PACKET_DEST_ID_SHIFT = 8,
};

enum {
    UCT_UD_PACKET_FLAG_AM      = UCS_BIT(0),
    UCT_UD_PACKET_FLAG_ACK_REQ = UCS_BIT(1),
    UCT_UD_PACKET_FLAG_ECN     = UCS_BIT(2),

    UCT_UD_PACKET_AM_ID_MASK   = UCS_MASK(UCT_UD_PACKET_AM_ID_SHIFT),
};

/*
network header layout

A - ack request 
E - explicit congestion notification (ecn)
N - negative acknoledgement

Active message packet header

31                                             8 7       3 2 1 0
+---------------------------------------------------------------+
|         dest_ep_id (24 bit)                   | am_id   |E|A|1| 
+---------------------------------------------------------------+
|       ack_psn (16 bit)        |           psn (16 bit)        |
+---------------------------------------------------------------+

Control packet header

31                                             8 7     4 3 2 1 0
+---------------------------------------------------------------+
|         dest_ep_id (24 bit)                   | reserv|N|E|A|0| 
+---------------------------------------------------------------+
|       ack_psn (16 bit)        |           psn (16 bit)        |
+---------------------------------------------------------------+

    // neth layout in human readable form 
    uint8_t            is_am:1;
    union {
        struct { // am false 
            uint8_t ack_req:1;
            uint8_t ecn:1;
            uint8_t nak:1;
            uint8_t reserved:4;
        } ctl;
        struct { // am true 
            uint8_t ack_req:1;
            uint8_t ecn:1;
            uint8_t am_id:5;
        } am;
    };
    uint32_t            dest_ep_id:24;
*/

typedef struct uct_ud_neth {
    uint32_t            packet_type;
    uct_ud_psn_t        psn;
    uct_ud_psn_t        ack_psn;
} UCS_S_PACKED uct_ud_neth_t;

typedef struct uct_ud_send_skb {
    ucs_callbackq_elem_t    queue;      /* in send window */
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

static inline uint32_t uct_ud_neth_get_dest_id(uct_ud_neth_t *neth)
{
    return neth->packet_type >> UCT_UD_PACKET_DEST_ID_SHIFT;
}
static inline void uct_ud_neth_set_dest_id(uct_ud_neth_t *neth, uint32_t id)
{
    neth->packet_type |= (id << UCT_UD_PACKET_DEST_ID_SHIFT);
}

static inline uint8_t uct_ud_neth_get_am_id(uct_ud_neth_t *neth)
{
    return neth->packet_type >> UCT_UD_PACKET_AM_ID_SHIFT;
}
static inline void uct_ud_neth_set_am_id(uct_ud_neth_t *neth, uint8_t id)
{
    neth->packet_type |= (id << UCT_UD_PACKET_AM_ID_SHIFT);
}

#endif

