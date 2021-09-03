/**
* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
*
* See file LICENSE for terms.
*/

#ifndef MLX5_ECE_H_
#define MLX5_ECE_H_

#include <ucs/config/types.h>

enum  {
    MLX5_ECE_DISABLED  = 0,
    MLX5_ECE_VER_1     = 1,
    MLX5_ECE_VER_SR    = 2,  //selective repeat
    MLX5_ECE_VER_MAX   = MLX5_ECE_VER_SR,
};

typedef struct mlx5_ece_cfg {
    ucs_on_off_auto_value_t ece_enable;
    ucs_on_off_auto_value_t ece_sr;
} mlx5_ece_cfg_t;

#endif
