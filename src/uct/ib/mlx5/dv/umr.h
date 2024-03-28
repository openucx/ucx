/**
* Copyright (C) 2023, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_UMR_H_
#define UCT_IB_MLX5_UMR_H_

#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/rc/base/rc_def.h>
#include <uct/api/uct_def.h>

typedef struct uct_ib_mlx5_umr *uct_ib_mlx5_umr_h;


ucs_status_t uct_ib_umr_init(uct_md_t *md, uct_ib_mlx5_umr_h *umr_p);


void uct_ib_umr_cleanup(uct_ib_mlx5_umr_h umr);


uct_ib_mlx5_txwq_t *uct_ib_umr_get_txwq(uct_ib_mlx5_umr_h umr);


ucs_status_t uct_ib_umr_post(uct_ib_mlx5_umr_h umr, void *wqe_end,
                             uint32_t lkey);

#endif
