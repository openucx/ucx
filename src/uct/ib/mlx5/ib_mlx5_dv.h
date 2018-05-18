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

typedef struct uct_ib_mlx5_qp_info {
    uint32_t           qpn;           /* QP number */
    volatile uint32_t  *dbrec;        /* QP doorbell record in RAM */

    struct {
            void       *buf;          /* Work queue buffer */
            unsigned   wqe_cnt;       /* Number of WQEs in the work queue */
            unsigned   stride;        /* Size of each WQE */
    } sq, rq;

    struct {
            void       *reg;          /* BlueFlame register */
            unsigned   size;          /* BlueFlame register size (0 - unsupported) */
    } bf;
} uct_ib_mlx5_qp_info_t;


typedef struct uct_ib_mlx5_srq_info {
    void               *buf;          /* SRQ queue buffer */
    volatile uint32_t  *dbrec;        /* SRQ doorbell record in RAM */
    unsigned           stride;        /* Size of each WQE */
    unsigned           head;
    unsigned           tail;
} uct_ib_mlx5_srq_info_t;

/* Completion queue */
typedef struct uct_ib_mlx5_cq {
    void               *cq_buf;
    unsigned           cq_ci;
    unsigned           cq_length;
    unsigned           cqe_size_log;
    unsigned           cq_num;
} uct_ib_mlx5_cq_t;

/**
 * Get internal QP information.
 */
ucs_status_t uct_ib_mlx5_get_qp_info(struct ibv_qp *qp,
                                     uct_ib_mlx5_qp_info_t *qp_info);

/**
 * Get internal SRQ information.
 */
ucs_status_t uct_ib_mlx5_get_srq_info(struct ibv_srq *srq,
                                      uct_ib_mlx5_srq_info_t *srq_info);

/**
 * Get internal CQ information.
 */
ucs_status_t uct_ib_mlx5_get_cq(struct ibv_cq *cq, uct_ib_mlx5_cq_t *mlx5_cq);

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

#endif
