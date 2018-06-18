/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5.h"
#include "ib_mlx5_log.h"
#include <ucs/arch/bitops.h>

#if HAVE_DECL_MLX5DV_INIT_OBJ
ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t type)
{
    int ret;

    ret = mlx5dv_init_obj(&obj->dv, type);
    if (ret != 0) {
        ucs_error("DV failed to get mlx5 information. Type %lx. "
                  "Please make sure the installed libmlx5 version matches "
                  "the one UCX was compiled with (%s)", type, LIB_MLX5_VER);
        return UCS_ERR_NO_DEVICE;
    }

    if (type & MLX5DV_OBJ_CQ) {
        uct_ib_mlx5dv_cq_t *cq;
        unsigned cqe_size;

        cq = ucs_container_of(obj->dv.cq.out, uct_ib_mlx5dv_cq_t, dv);
        cq->cq_ci     = 0;
        cqe_size      = cq->dv.cqe_size;

        /* Move buffer forward for 128b CQE, so we would get pointer to the 2nd
         * 64b when polling.
         */
        cq->dv.buf += cqe_size - sizeof(struct mlx5_cqe64);

        ret = ibv_exp_cq_ignore_overrun(obj->dv.cq.in);
        if (ret != 0) {
            ucs_error("Failed to modify send CQ to ignore overrun: %s",
                      strerror(ret));
            return UCS_ERR_UNSUPPORTED;
        }

        cq->cqe_size_log = ucs_ilog2(cqe_size);
        ucs_assert_always((1 << cq->cqe_size_log) == cqe_size);

        cq->dv.uar = uct_dv_get_info_uar0(cq->dv.uar);
    }

    return UCS_OK;
}
#endif

enum {
    UCT_IB_MLX5_CQ_SET_CI  = 0,
    UCT_IB_MLX5_CQ_ARM_DB  = 1,
};

int uct_dv_mlx5_arm_cq(uct_ib_mlx5dv_cq_t *cq, int solicited)
{
    uint64_t doorbell;
    uint32_t sn;
    uint32_t ci;
    uint32_t cmd;

    sn  = cq->cq_ci & 3;
    ci  = cq->cq_ci & 0xffffff;
    cmd = solicited ? MLX5_CQ_DB_REQ_NOT_SOL : MLX5_CQ_DB_REQ_NOT;

    cq->dv.dbrec[UCT_IB_MLX5_CQ_ARM_DB] = htobe32(sn << 28 | cmd | ci);

    ucs_memory_cpu_fence();

    doorbell = sn << 28 | cmd | ci;
    doorbell <<= 32;
    doorbell |= cq->dv.cqn;

    *(uint64_t *)((uint8_t *)cq->dv.uar + 0x20) = htobe64(doorbell);

    ucs_memory_bus_store_fence();

    return 0;
}
