/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef UD_MLX5_H
#define UD_MLX5_H

#include "ud_mlx5_common.h"

#include <uct/ib/ud/base/ud_iface.h>
#include <uct/ib/ud/base/ud_ep.h>


typedef struct {
    uct_ud_ep_t                         super;
    uct_ib_mlx5_base_av_t               av;
    uint8_t                             is_global;
    struct mlx5_grh_av                  grh_av;
} uct_ud_mlx5_ep_t;


typedef struct {
    uct_ud_iface_config_t               super;
    uct_ud_mlx5_iface_common_config_t   mlx5_common;
} uct_ud_mlx5_iface_config_t;


typedef struct {
    uct_ud_iface_t                      super;
    struct {
        uct_ib_mlx5_txwq_t              wq;
        uct_ib_mlx5_cq_t                cq;
    } tx;
    struct {
        uct_ib_mlx5_rxwq_t              wq;
        uct_ib_mlx5_cq_t                cq;
    } rx;
    uct_ud_mlx5_iface_common_t          mlx5_common;
} uct_ud_mlx5_iface_t;


static inline unsigned uct_ud_mlx5_tx_moderation(uct_ud_mlx5_iface_t *iface)
{
    if (iface->super.tx.unsignaled >= UCT_UD_TX_MODERATION) {
        iface->super.tx.unsignaled = 0;
        return MLX5_WQE_CTRL_CQ_UPDATE;
    }
    iface->super.tx.unsignaled++;
    return 0;
}

#endif

