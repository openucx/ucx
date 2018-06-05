/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5.h"
#include "ib_mlx5_log.h"

#if HAVE_DECL_MLX5DV_INIT_OBJ
ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t type)
{
    int ret;

    ret = mlx5dv_init_obj(&obj->dv, type);
    if (ret != 0) {
        ucs_error("DV failed to get mlx5 information. Type %lx.", type);
        return UCS_ERR_NO_DEVICE;
    }

    if (type & MLX5DV_OBJ_CQ) {
        /* enable cq_clean in libmlx5 */
        uct_ib_mlx5_cq_set_flags(obj->dv.cq.in, 0);
    }

    return UCS_OK;
}
#endif

enum {
    UCT_IB_MLX5_CQ_SET_CI  = 0,
    UCT_IB_MLX5_CQ_ARM_DB  = 1,
};

int uct_ib_mlx5dv_arm_cq(uct_ib_mlx5_cq_t *cq, int solicited)
{
    uint64_t doorbell;
    uint32_t sn;
    uint32_t ci;
    uint32_t cmd;

    sn  = cq->cq_ci & 3;
    ci  = cq->cq_ci & 0xffffff;
    cmd = solicited ? MLX5_CQ_DB_REQ_NOT_SOL : MLX5_CQ_DB_REQ_NOT;

    cq->dbrec[UCT_IB_MLX5_CQ_ARM_DB] = htobe32(sn << 28 | cmd | ci);

    ucs_memory_cpu_fence();

    doorbell = sn << 28 | cmd | ci;
    doorbell <<= 32;
    doorbell |= cq->cq_num;

    *(uint64_t *)((uint8_t *)cq->uar + MLX5_CQ_DOORBELL) = htobe64(doorbell);

    ucs_memory_bus_store_fence();

    return 0;
}
