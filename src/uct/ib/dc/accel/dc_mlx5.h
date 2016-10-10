/**
* Copyright (C) Mellanox Technologies Ltd. 2016-.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_MLX5_H
#define UCT_DC_MLX5_H

#include <uct/ib/dc/base/dc_iface.h>
#include <uct/ib/dc/base/dc_ep.h>
#include <uct/ib/rc/accel/rc_mlx5_common.h>
#include <uct/ib/ud/accel/ud_mlx5_common.h>


typedef struct uct_dc_mlx5_iface_config {
    uct_dc_iface_config_t               super;
    uct_ud_mlx5_iface_common_config_t   ud_common;
} uct_dc_mlx5_iface_config_t;


typedef struct uct_dc_mlx5_iface {
    uct_dc_iface_t                      super;
    uct_rc_mlx5_iface_common_t          mlx5_common;
    uct_ud_mlx5_iface_common_t          ud_common;
    uct_ib_mlx5_txwq_t                  dci_wqs[UCT_DC_IFACE_MAX_DCIS];

} uct_dc_mlx5_iface_t;


typedef struct uct_dc_mlx5_ep {
    uct_dc_ep_t                         super;
    uct_ib_mlx5_base_av_t               av;
} uct_dc_mlx5_ep_t;


#endif
