/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UD_MLX5_COMMON_H_
#define UD_MLX5_COMMON_H_

#include <uct/ib/mlx5/ib_mlx5.h>


typedef struct uct_ud_mlx5_iface_common_config {
    int                          enable_compact_av;
} uct_ud_mlx5_iface_common_config_t;


typedef struct uct_ud_mlx5_iface_common {
    struct {
        int                      compact_av;
    } config;
} uct_ud_mlx5_iface_common_t;


extern ucs_config_field_t uct_ud_mlx5_iface_common_config_table[];


ucs_status_t uct_ud_mlx5_iface_common_init(uct_ib_iface_t *ib_iface,
                                           uct_ud_mlx5_iface_common_t *iface,
                                           uct_ud_mlx5_iface_common_config_t *config);


ucs_status_t
uct_ud_mlx5_iface_get_av(uct_ib_iface_t *iface,
                         uct_ud_mlx5_iface_common_t *ud_common_iface,
                         const uct_ib_address_t *ib_addr, unsigned path_index,
                         const char *usage, uct_ib_mlx5_base_av_t *base_av,
                         struct mlx5_grh_av *grh_av, int *is_global);

#endif
