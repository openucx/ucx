/** 
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*   
* See file LICENSE for terms.
*/
    
#ifndef UD_DEF_H_
#define UD_DEF_H_


#include <ucs/sys/math.h>

#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/frag_list.h>
#include <uct/ib/base/ib_iface.h>

#define UCT_UD_QP_HASH_SIZE     256    
#define UCT_UD_MAX_SGE          2
#define UCT_UD_TX_MODERATION    64
#define UCT_UD_MIN_INLINE       48
#define UCT_UD_HASH_SIZE        997

/* congestion avoidance settings */
/* UD uses additive increase/multiplicative decrease algorightm
 * See https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease
 *
 * tx window is increased when ack is received and decreased when
 * resend is scheduled. Ack must be a 'new' one that is it must
 * acknoledge packets on window. Increasing window on ack does not casue 
 * exponential window increase because, unlike tcp, only 2 packets 
 * per window are sent.
 *
 * Todo: 
 *
 * Consider trigering window decrease before resend timeout:
 * - on ECN (explicit congestion notification) from receiever. ECN can
 *   be based on some heuristic. For example on number of rx completions
 *   that receiver picked from CQ.
 * - upon receiving a 'duplicate ack' packet
 *
 * Consider using other algorithm (ex BIC/CUBIC)
 */
#define UCT_UD_CA_AI_VALUE      1   /* window += AI_VALUE */
#define UCT_UD_CA_MD_FACTOR     2   /* window = window/factor */
#define UCT_UD_CA_DUP_ACK_CNT   2   /* TODO: not implemented yet */

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

enum {
    UCT_UD_SEND_SKB_FLAG_ACK_REQ = UCS_BIT(1)
};

typedef struct uct_ud_send_skb {
    ucs_queue_elem_t        queue;      /* in send window */
    uint32_t                lkey;
    uint16_t                len;        /* data size */
    uint16_t                flags;
    uct_ud_neth_t           neth[0];
} UCS_S_PACKED uct_ud_send_skb_t;

typedef struct uct_ud_send_skb_inl {
    uct_ud_send_skb_t  super;
    uct_ud_neth_t      neth;
} UCS_S_PACKED uct_ud_send_skb_inl_t;

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

#endif

