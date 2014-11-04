/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_MLX5_H_
#define UCT_IB_MLX5_H_


#include <uct/api/uct_def.h>
#include <ucs/type/status.h>
#include <infiniband/mlx5_hw.h>
#include <infiniband/arch.h>
#include <netinet/in.h>


typedef struct uct_ib_mlx5_qp_info {
    uint32_t           qpn;           /* QP number */
    uint32_t           *dbrec;        /* QP doorbell record in RAM */

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


typedef struct uct_ib_mlx5_cq_info {
    uint32_t           cqn;           /* CQ number */
    unsigned           cqe_cnt;       /* Number of CQEs in the queue */
    void               *buf;          /* CQ buffer */
    uint32_t           *dbrec;        /* CQ doorbell record */
    unsigned           cqe_size;      /* Size of a CQE */
} uct_ib_mlx5_cq_info_t;


/**
 * Get internal QP information.
 */
ucs_status_t uct_ib_mlx5_get_qp_info(struct ibv_qp *qp, uct_ib_mlx5_qp_info_t *qp_info);

/**
 * Get internal CQ information.
 */
ucs_status_t uct_ib_mlx5_get_cq_info(struct ibv_cq *cq, uct_ib_mlx5_cq_info_t *cq_info);

/**
 * Update CI to support req_notify_cq
 */
void uct_ib_mlx5_update_cq_ci(struct ibv_cq *cq, unsigned cq_ci);

/**
 * Get internal AV information.
 */
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av);


#endif
