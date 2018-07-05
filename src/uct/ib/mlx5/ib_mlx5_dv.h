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

#endif
