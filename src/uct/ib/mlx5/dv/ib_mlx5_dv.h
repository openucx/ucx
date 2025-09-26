/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_DV_H_
#define UCT_IB_MLX5_DV_H_

#include <ucs/type/status.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

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

typedef struct uct_ib_mlx5dv_qp_tmp_objs {
    struct ibv_srq *srq;
    struct ibv_cq  *cq;
} uct_ib_mlx5dv_qp_tmp_objs_t;


/**
 * Get internal verbs information.
 */
ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t type);

/**
 * Initialize DC-specific DV QP attributes.
 */
void uct_ib_mlx5dv_dc_qp_init_attr(struct mlx5dv_qp_init_attr *dv_attr,
                                   enum mlx5dv_dc_type dc_type);

/**
 * Initialize DCT-specific QP attributes.
 */
void uct_ib_mlx5dv_dct_qp_init_attr(uct_ib_qp_init_attr_t *qp_attr,
                                    struct mlx5dv_qp_init_attr *dv_attr,
                                    struct ibv_pd *pd, struct ibv_cq *cq,
                                    struct ibv_srq *srq);

/**
 * Creates CQ and SRQ which are needed for creating QP.
 */
ucs_status_t
uct_ib_mlx5dv_qp_tmp_objs_create(struct ibv_pd *pd,
                                 uct_ib_mlx5dv_qp_tmp_objs_t *qp_tmp_objs,
                                 int silent);

/**
 * Closes CQ and SRQ which are needed for creating QP.
 */
void uct_ib_mlx5dv_qp_tmp_objs_destroy(uct_ib_mlx5dv_qp_tmp_objs_t *qp_tmp_objs);

/**
 * Calculates TX WQE ratio.
 */
size_t uct_ib_mlx5dv_calc_tx_wqe_ratio(struct ibv_qp *qp, uint32_t max_send_wr,
                                       size_t *tx_wqe_ratio_p);

/**
 * Sets QP initialization attributes.
 */
void uct_ib_mlx5dv_qp_init_attr(uct_ib_qp_init_attr_t *qp_init_attr,
                                struct ibv_pd *pd,
                                const uct_ib_mlx5dv_qp_tmp_objs_t *qp_tmp_objs,
                                enum ibv_qp_type qp_type, uint32_t max_recv_wr);

/**
 * Get internal AV information.
 */
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av);

void *uct_dv_get_info_uar0(void *uar);

#endif
