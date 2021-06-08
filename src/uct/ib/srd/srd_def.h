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
#define UCT_SRD_SEND_DESC_ALIGN  UCS_SYS_CACHE_LINE_SIZE
#define UCT_SRD_SEND_OP_ALIGN    UCS_SYS_CACHE_LINE_SIZE


typedef ucs_frag_list_sn_t          uct_srd_psn_t;
typedef struct uct_srd_iface        uct_srd_iface_t;
typedef struct uct_srd_iface_addr   uct_srd_iface_addr_t;
typedef struct uct_srd_iface_peer   uct_srd_iface_peer_t;
typedef struct uct_srd_ep           uct_srd_ep_t;
typedef struct uct_srd_ep_addr      uct_srd_ep_addr_t;
typedef struct uct_srd_send_op      uct_srd_send_op_t;
typedef struct uct_srd_send_desc    uct_srd_send_desc_t;
typedef struct uct_srd_ctl_hdr      uct_srd_ctl_hdr_t;


enum {
    UCT_SRD_PACKET_DEST_ID_SHIFT   = 24,
    UCT_SRD_PACKET_AM_ID_SHIFT     = 27,
};

enum {
    UCT_SRD_PACKET_FLAG_AM        = UCS_BIT(24),
    UCT_SRD_PACKET_FLAG_PUT       = UCS_BIT(25),
    UCT_SRD_PACKET_FLAG_CTLX      = UCS_BIT(26),

    /* Pure credit grant: empty control message indicating credit grant */
    UCT_SRD_PACKET_FLAG_FC_PGRANT = UCS_BIT(27),

    UCT_SRD_PACKET_AM_ID_MASK     = UCS_MASK(UCT_SRD_PACKET_AM_ID_SHIFT),
    UCT_SRD_PACKET_DEST_ID_MASK   = UCS_MASK(UCT_SRD_PACKET_DEST_ID_SHIFT),
};

enum {
    UCT_SRD_PACKET_CREQ = 1,
    UCT_SRD_PACKET_CREP = 2,
};

/* Used for the fc member in uct_srd_neth */
enum {
    /* Piggy-backed credit Grant: ep should update its FC wnd as soon as iteceives AM with
     * this bit set. Can be bundled with either soft or hard request bits */
    UCT_SRD_PACKET_FLAG_FC_GRANT = UCS_BIT(0),

    /* Soft Credit Request: indicates that receiving peer needs to piggy-back credits
     * grant to counter AM (if any). Can be bundled with
     * UCT_SRD_PACKET_FLAG_FC_GRANT  */
    UCT_SRD_PACKET_FLAG_FC_SREQ  = UCS_BIT(1),

    /* Hard Credit Request: indicates that sender wnd is close to be exhausted.
     * The receiving peer must send a pure fc grant control message as soon as it
     * receives AM  with this bit set. Can be bundled with
     * UCT_SRD_PACKET_FLAG_FC_GRANT */
    UCT_SRD_PACKET_FLAG_FC_HREQ  = UCS_BIT(2),
};

/*
network header layout

Active message packet header
G - piggy-backed fc grant
H - fc hard request
S - fc soft request

+---------------------------------------------------------------+
| am_id |rsv|1|              dest_ep_id (24 bit)                |
+---------------------------------------------------------------+
|  rsv  |H|S|G|      psn (16 bit)     |
+--------------------------------------

Put emulation packet header

+---------------------------------------------------------------+
|   rsv   |1|0|              dest_ep_id (24 bit)                |
+---------------------------------------------------------------+
|  rsv  |H|S|G|      psn (16 bit)     |
+--------------------------------------

Control packet header
C - control packet extended header (CREQ/CREP)
G - fc pure grant

+---------------------------------------------------------------+
| rsv |G|C|0|0|              dest_ep_id (24 bit)                |
+---------------------------------------------------------------+
|     rsv     |      psn (16 bit)     |
+--------------------------------------

    // neth layout in human readable form
    uint32_t            dest_ep_id:24;
    uint8_t             is_am:1;
    uint8_t             is_put:1;
    union {
        struct { // am true
            uint8_t     reserved:2;
            uint8_t     am_id:5;
        } am;
        union { // am false
            struct { // put true
                uint8_t reserved:6;
            } put;
            struct { // put false
                uint8_t ctlx:1;
                uint8_t fc_grant_pure:1;
                uint8_t reserved:4;
            } ctl;
        }
    };
    union {
        struct { // am true
            uint8_t fc_grant:1;
            uint8_t fc_soft_req:1;
            uint8_t fc_hard_req:1;
            uint8_t reserved:5;
        } am;
        struct { // am false
            uint8_t reserved:8;
        } ctl;
    };
*/

typedef struct uct_srd_neth {
    uint32_t             packet_type;
    uint8_t              fc;
    uct_srd_psn_t        psn;
} UCS_S_PACKED uct_srd_neth_t;


enum {
    UCT_SRD_SEND_OP_FLAG_FLUSH   = UCS_BIT(0), /* dummy send op for flush */
    UCT_SRD_SEND_OP_FLAG_RMA     = UCS_BIT(1), /* send op is for an RMA op */
    UCT_SRD_SEND_OP_FLAG_PURGED  = UCS_BIT(2), /* send op has been purged */

#if UCS_ENABLE_ASSERT
    UCT_SRD_SEND_OP_FLAG_INVALID = UCS_BIT(7), /* send op has been released */
#else
    UCT_SRD_SEND_OP_FLAG_INVALID = 0,
#endif
};


typedef void (*uct_srd_send_op_comp_handler_t)(uct_srd_send_op_t *send_op);


/*
 * Send descriptor
 * - used for any send op (including RDMA READ) that
 *   requires some form of handling after completion
 */
struct uct_srd_send_op {
    /* link in ep outstanding send queue */
    ucs_queue_elem_t                 out_queue;

    /* number of bytes that should be sent/received */
    uint32_t                         len;
    uint16_t                         flags;

    /* ep that does the send */
    uct_srd_ep_t                     *ep;

    uct_completion_t                 *user_comp;

    /* handler that is called at send completion time */
    uct_srd_send_op_comp_handler_t   comp_handler;
} UCS_V_ALIGNED(UCT_SRD_SEND_OP_ALIGN);


/*
 * - for both am bcopy and am zcopy, network comes after the descriptor
 * - for am zcopy, the am header comes after the network header
 * - for am bcopy, the copied data comes after the network header
 * - for CREQ/CREP, control header comes after the network header
 * - for get bcopy, the copied data comes after the descriptor
 *   am short and get zcopy do not use this struct. They use send_op only
 */
struct uct_srd_send_desc {
    uct_srd_send_op_t                super;
    uct_unpack_callback_t            unpack_cb;
    void                             *unpack_arg;
    uint32_t                         lkey;
} UCS_S_PACKED UCS_V_ALIGNED(UCT_SRD_SEND_DESC_ALIGN);


typedef struct uct_srd_recv_desc {
    uct_ib_iface_recv_desc_t super;
    struct {
        ucs_frag_list_elem_t     elem;
    } ooo;
    uint32_t                     data_len;
} uct_srd_recv_desc_t;


typedef struct uct_srd_am_short_hdr {
    uct_srd_neth_t neth;
    uint64_t       hdr;
} UCS_S_PACKED uct_srd_am_short_hdr_t;


typedef struct uct_srd_put_hdr {
    uct_srd_neth_t neth;
    uint64_t       rva;
} UCS_S_PACKED uct_srd_put_hdr_t;


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

static inline uct_srd_neth_t*
uct_srd_send_desc_neth(uct_srd_send_desc_t *desc)
{
    return (uct_srd_neth_t*)(desc + 1);
}

static inline uint32_t
uct_srd_neth_is_am(uct_srd_neth_t *neth)
{
    return neth->packet_type & UCT_SRD_PACKET_FLAG_AM;
}

static inline uint32_t
uct_srd_neth_is_pure_grant(uct_srd_neth_t *neth)
{
    return neth->packet_type & UCT_SRD_PACKET_FLAG_FC_PGRANT;
}

static inline uint32_t
uct_srd_neth_is_put(uct_srd_neth_t *neth)
{
    return neth->packet_type & UCT_SRD_PACKET_FLAG_PUT;
}

#endif
