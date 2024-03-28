/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_GGA_MLX5_H_
#define UCT_GGA_MLX5_H_

#if HAVE_DEVX

#include <uct/ib/base/ib_md.h>


ucs_status_t uct_ib_mlx5_gga_md_open(struct ibv_device *ibv_device,
                                     const uct_ib_md_config_t *md_config,
                                     struct uct_ib_md **md_p);

#else /* HAVE_DEVX */

static inline ucs_status_t
uct_ib_mlx5_gga_md_open(struct ibv_device *ibv_device,
                        const uct_ib_md_config_t *md_config,
                        struct uct_ib_md **md_p)
{
    return UCS_ERR_UNSUPPORTED;
}

#endif /* HAVE_DEVX */

#endif /* UCT_GGA_MLX5_H_ */
