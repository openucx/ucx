/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_IFACE_H
#define UCT_IB_IFACE_H

#include "ib_device.h"

#include <uct/api/uct.h>
#include <uct/tl/tl_base.h>
#include <ucs/sys/compiler.h>
#include <ucs/config/parser.h>


typedef struct uct_ib_iface_addr {
    uct_iface_addr_t    super;
    uint16_t            lid; /* TODO support RoCE/GRH */
} uct_ib_iface_addr_t;


typedef struct uct_ib_iface {
    uct_base_iface_t        super;
    uint8_t                 port_num;
    /* TODO
     * lmc
     * comp_channel;
     */
    unsigned                gid_index;
    unsigned                sl;
    uct_ib_iface_addr_t     addr;
    struct ibv_cq           *send_cq;
    struct ibv_cq           *recv_cq;

    struct {
        unsigned            rx_headroom;         /* user-requested headroom */
        unsigned            rx_payload_offset;   /* offset from desc to payload */
        unsigned            rx_hdr_offset;       /* offset from desc to network header */
        unsigned            seg_size;
    } config;

} uct_ib_iface_t;

typedef struct uct_ib_iface_config {
    uct_iface_config_t      super;

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many fragments can be batched to one post send */
        size_t              min_inline;      /* Inline space to reserve */
        unsigned            min_sge;         /* How many SG entries to support */
        unsigned            cq_moderation;   /* How many TX messages are batched to one CQE */
        uct_iface_mpool_config_t mp;
    } tx;

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many buffers can be batched to one post receuive */
        size_t              inl;             /* Inline space to reserve in CQ/QP */
        uct_iface_mpool_config_t mp;
    } rx;

    /* IB GID index to use  */
    unsigned                gid_index;

    /* IB SL to use */
    unsigned                sl;

} uct_ib_iface_config_t;


/*
 * The offset to the payload is the maximum between user-requested headroom
 * and transport-specific data/header. When the active message callback is invoked,
 * it gets a pointer to the beginning of the headroom.
 * The headroom can be either smaller (1) or larger (2) than the transport data.
 *
 * (1)
 *
 *                   am_callback
 *                   |
 * +------+----------+-----------+---------+
 * | LKey |   ???    | Head Room | Payload |
 * +------+----------+--+--------+---------+
 * | LKey |     TL data | TL hdr | Payload |
 * +------+-------------+--------+---------+
 *                      |
 *                      post_receive
 *
 * (2)
 *        am_callback
 *        |
 * +------+----------------------+---------+
 * | LKey |      Head Room       | Payload |
 * +------+---------+---+--------+---------+
 * | LKey | TL data | ? | TL hdr | Payload |
 * +------+---------+---+--------+---------+
 *                      |
 *                      post_receive
 *
 *        <----- rx_headroom ---->
 * <------- rx_payload_offset --->
 * <--- rx_hdr_offset -->
 *
 */
typedef struct uct_ib_iface_recv_desc {
    uint32_t                lkey;
} UCS_S_PACKED uct_ib_iface_recv_desc_t;



extern ucs_config_field_t uct_ib_iface_config_table[];


/**
 * Create memory pool of receive descriptors.
 */
ucs_status_t uct_ib_iface_recv_mpool_create(uct_ib_iface_t *iface,
                                            uct_ib_iface_config_t *config,
                                            const char *name, ucs_mpool_h *mp_p);

static inline uct_ib_device_t * uct_ib_iface_device(uct_ib_iface_t *iface)
{
    return ucs_derived_of(iface->super.super.pd, uct_ib_device_t);
}

static inline struct ibv_exp_port_attr* uct_ib_iface_port_attr(uct_ib_iface_t *iface)
{
    return uct_ib_device_port_attr(uct_ib_iface_device(iface), iface->port_num);
}

static inline void* uct_ib_iface_recv_desc_hdr(uct_ib_iface_t *iface,
                                               uct_ib_iface_recv_desc_t *desc)
{
    return (void*)desc + iface->config.rx_hdr_offset;
}

typedef struct uct_ib_recv_wr {
    struct ibv_recv_wr ibwr;
    struct ibv_sge     sg;
} uct_ib_recv_wr_t; 

/**
 * prepare a list of n work requests that can be passed to
 * ibv_post_recv()
 *
 * @return number of prepared wrs
 */
int uct_ib_iface_prepare_rx_wrs(uct_ib_iface_t *iface,
                                ucs_mpool_h rx_mp, uct_ib_recv_wr_t *wrs, unsigned n);


#endif
