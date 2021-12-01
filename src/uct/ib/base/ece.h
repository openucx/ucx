/**
* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
*
* See file LICENSE for terms.
*/

#ifndef MLX5_ECE_H_
#define MLX5_ECE_H_

#include <stdint.h>
#include <infiniband/verbs.h>
#include <ucs/config/types.h>
#include <ucs/debug/assert.h>


enum  {
    MLX5_ECE_DISABLED  = 0,
    MLX5_ECE_VER_1     = 1,
    MLX5_ECE_VER_2     = 2,  /* selective repeat */
    MLX5_ECE_VER_3     = 3,  /* congestion control */
    MLX5_ECE_VER_MAX   = MLX5_ECE_VER_3,
};


/* user ece configuration option */
typedef struct mlx5_ece_cfg {
    ucs_on_off_auto_value_t enable;
    ucs_on_off_auto_value_t sr;
} mlx5_ece_cfg_t;


union ece_t {
    struct {
        uint32_t sr  : 1;
        uint32_t cc  : 8;
        uint32_t rsv : 19;
        uint32_t ver : 4;
    } field;

    uint32_t val;
};


#define ECE_USED_BITS (MLX5_ECE_VER_MAX << 28 | 0x1fe | 0x1)


/* ece configuration under user cfg and hardware limitation*/
typedef struct mlx5_ece {
    uint8_t     enable;

    union ece_t ece;
} mlx5_ece_t;


static UCS_F_ALWAYS_INLINE
uint32_t ece_intersect(uint32_t val0, uint32_t val1)
{
    union ece_t ece0, ece1, ece_rst;
    uint32_t cc_algo_bit;

    ece0.val    = val0;
    ece1.val    = val1;
    ece_rst.val = 0;

    ucs_assert(ece0.field.ver != 1 && ece1.field.ver != 1);

    if (ece0.field.ver == 0 || ece1.field.ver == 0) {
        return ece_rst.val;
    }

    /* selective repeat */
    ece_rst.field.sr = ece0.field.sr & ece1.field.sr;

    /* congestion control */
    cc_algo_bit = ffs(ece0.field.cc & ece1.field.cc);
    ece_rst.field.cc = cc_algo_bit ? (1 << (cc_algo_bit - 1)) : 0;

    /* ece version */
    ece_rst.field.ver = MLX5_ECE_VER_MAX;

    return ece_rst.val;
}

#endif
