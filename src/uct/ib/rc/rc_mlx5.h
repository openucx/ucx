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

#include <ucs/type/class.h>
#include <uct/ib/mlx5/ib_mlx5.h>


/**
 * RC mlx5 interface configuration
 */
typedef struct uct_rc_mlx5_iface_config {
    uct_rc_iface_config_t  super;
    /* TODO wc_mode, UAR mode SnB W/A... */
} uct_rc_mlx5_iface_config_t;


/**
 * RC Communication context for mlx5
 */
typedef struct uct_rc_mlx5_iface {
    uct_rc_ep_t      super;
    unsigned         qpn_ds;

    struct {
        unsigned       sw_pi;
        unsigned       max_pi;
        void           *seg;
        void           *bf_reg;
        unsigned long  bf_size;
        uint32_t       *dbrec;
        void           *qstart;
        void           *qend;
    } tx;
} uct_rc_mlx5_ep_t;


/**
 * Remote RC endpoint for mlx5
 */
typedef struct {
    uct_rc_iface_t     super;
    struct {
        void           *cq_buf;
        unsigned       cq_ci;
        unsigned       cq_length;
    } tx;
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

#endif
