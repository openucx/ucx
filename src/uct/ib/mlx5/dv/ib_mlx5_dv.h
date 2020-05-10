/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_DV_H_
#define UCT_IB_MLX5_DV_H_

#ifndef UCT_IB_MLX5_H_
#  error "Never include <uct/ib/mlx5/ib_mlx5_dv.h> directly; use <uct/ib/mlx5/ib_mlx5.h> instead."
#endif

#include <ucs/type/status.h>
#include <infiniband/verbs.h>

typedef struct {
    struct mlx5dv_obj  dv;
#ifdef HAVE_IBV_EXP_DM
    struct {
        struct ibv_exp_dm   *in;
        struct mlx5dv_dm    *out;
    } dv_dm;
#endif
} uct_ib_mlx5dv_t;

typedef struct {
    struct mlx5dv_qp   dv;
} uct_ib_mlx5dv_qp_t;

typedef struct {
    struct mlx5dv_srq  dv;
} uct_ib_mlx5dv_srq_t;

/* Completion queue */
typedef struct {
    struct mlx5dv_cq   dv;
} uct_ib_mlx5dv_cq_t;

/**
 * Get internal verbs information.
 */
ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t type);

/**
 * Update CI to support req_notify_cq
 */
void uct_ib_mlx5_update_cq_ci(struct ibv_cq *cq, unsigned cq_ci);

/**
 * Retrieve CI from the driver
 */
unsigned uct_ib_mlx5_get_cq_ci(struct ibv_cq *cq);

/**
 * Get internal AV information.
 */
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av);

/**
 * Backports for legacy bare-metal support
 */
struct ibv_qp *uct_dv_get_cmd_qp(struct ibv_srq *srq);

void *uct_dv_get_info_uar0(void *uar);

/*
 * DM backports
 */
#ifdef HAVE_IBV_EXP_DM
#  define ibv_dm            ibv_exp_dm
#  define ibv_alloc_dm_attr ibv_exp_alloc_dm_attr
#  define ibv_alloc_dm      ibv_exp_alloc_dm
#  define ibv_free_dm       ibv_exp_free_dm

struct mlx5dv_dm {
    void                *buf;
    uint64_t            length;
    uint64_t            comp_mask;
};

enum {
    MLX5DV_OBJ_DM   = 1 << 4,
};

static struct ibv_mr * UCS_F_MAYBE_UNUSED
ibv_reg_dm_mr(struct ibv_pd *pd, struct ibv_dm *dm,
              uint64_t dm_offset, size_t length, unsigned int access_flags)
{
    struct ibv_exp_reg_mr_in mr_in = {};
    mr_in.pd        = pd;
    mr_in.comp_mask = IBV_EXP_REG_MR_DM;
    mr_in.dm        = dm;
    mr_in.length    = length;
    return ibv_exp_reg_mr(&mr_in);
}

typedef struct uct_mlx5_dm_va {
    struct ibv_dm      ibv_dm;
    size_t             length;
    uint64_t           *start_va;
} uct_mlx5_dm_va_t;

static ucs_status_t UCS_F_MAYBE_UNUSED
uct_ib_mlx5_get_dm_info(struct ibv_exp_dm *dm, struct mlx5dv_dm *dm_info)
{
    dm_info->buf = ((uct_mlx5_dm_va_t*)dm)->start_va;
    return UCS_OK;
}

# define UCT_IB_MLX5_DV_DM(_obj) _obj.dv_dm
#else
# define UCT_IB_MLX5_DV_DM(_obj) _obj.dv.dm
#endif

#endif
