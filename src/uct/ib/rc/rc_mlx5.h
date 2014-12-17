/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_MLX5_H
#define UCT_RC_MLX5_H

#include "rc_iface.h"
#include "rc_ep.h"

#include <uct/ib/mlx5/ib_mlx5.h>
#include <ucs/datastruct/queue.h>
#include <ucs/type/class.h>


typedef struct uct_rc_mlx5_recv_desc uct_rc_mlx5_recv_desc_t;
struct uct_rc_mlx5_recv_desc {
    uct_ib_iface_recv_desc_t   super;
    ucs_queue_elem_t           queue;
} UCS_S_PACKED;


/**
 * RC mlx5 interface configuration
 */
typedef struct uct_rc_mlx5_iface_config {
    uct_rc_iface_config_t  super;
    /* TODO wc_mode, UAR mode SnB W/A... */
} uct_rc_mlx5_iface_config_t;


/**
 * RC remote endpoint
 */
typedef struct uct_rc_mlx5_ep {
    uct_rc_ep_t      super;
    unsigned         qpn_ds;

    struct {
        uint16_t       sw_pi;
        uint16_t       max_pi;
        uint16_t       wqe_cnt;
        void           *seg;
        void           *bf_reg;
        unsigned long  bf_size;
        uint32_t       *dbrec;
        void           *qstart;
        void           *qend;
    } tx;
} uct_rc_mlx5_ep_t;


/**
 * RC communication interface
 */
typedef struct {
    uct_rc_iface_t         super;

    struct {
        uct_ib_mlx5_cq_t   cq;
    } tx;

    struct {
        uct_ib_mlx5_cq_t   cq;
        ucs_queue_head_t   desc_q;
        void               *buf;
        uint32_t           *db;
        uint16_t           head;
        uint16_t           tail;
        uint16_t           sw_pi;
    } rx;

} uct_rc_mlx5_iface_t;


/**
 * RC/RDMA Inline WQE segment
 */
typedef struct {
    struct mlx5_wqe_ctrl_seg     ctrl;
    struct mlx5_wqe_raddr_seg    raddr;
    struct mlx5_wqe_inl_data_seg inl;
} UCS_S_PACKED uct_rc_mlx5_wqe_rdma_inl_seg_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_mlx5_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_mlx5_ep_t, uct_ep_t);

ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, void *buffer, unsigned length,
                                      uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     void *buffer, unsigned length);

ucs_status_t uct_rc_mlx5_ep_flush(uct_ep_h tl_ep);

#endif
