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


#define UCT_IB_MLX5_CQE64_MAX_INL   32 /* Inline scatter size in 64-byte CQE */
#define UCT_IB_MLX5_CQE128_MAX_INL  64 /* Inline scatter size in 128-byte CQE */


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


typedef struct uct_ib_mlx5_srq_info {
    void               *buf;          /* SRQ queue buffer */
    uint32_t           *dbrec;        /* SRQ doorbell record in RAM */
    unsigned           stride;        /* Size of each WQE */
    unsigned           head;
    unsigned           tail;
} uct_ib_mlx5_srq_info_t;


typedef struct uct_ib_mlx5_cq {
    void               *cq_buf;
    unsigned           cq_ci;
    unsigned           cq_length;
    unsigned           cqe_size;
} uct_ib_mlx5_cq_t;



/**
 * Get internal QP information.
 */
ucs_status_t uct_ib_mlx5_get_qp_info(struct ibv_qp *qp, uct_ib_mlx5_qp_info_t *qp_info);

/**
 * Get internal SRQ information.
 */
ucs_status_t uct_ib_mlx5_get_srq_info(struct ibv_srq *srq, uct_ib_mlx5_srq_info_t *srq_info);

/**
 * Get internal CQ information.
 */
ucs_status_t uct_ib_mlx5_get_cq(struct ibv_cq *cq, uct_ib_mlx5_cq_t *mlx5_cq);

/**
 * Update CI to support req_notify_cq
 */
void uct_ib_mlx5_update_cq_ci(struct ibv_cq *cq, unsigned cq_ci);

/**
 * Get internal AV information.
 */
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av);

void uct_ib_mlx5_check_completion(struct mlx5_cqe64 *cqe);


static inline void uct_ib_mlx5_wqe_set_data_seg(struct mlx5_wqe_data_seg *seg,
                                                void *ptr, unsigned length,
                                                uint32_t lkey)
{
    seg->byte_count = htonl(length);
    seg->lkey       = htonl(lkey);
    seg->addr       = htonll((uintptr_t)ptr);
}

static inline struct mlx5_cqe64* uct_ib_mlx5_get_cqe(uct_ib_mlx5_cq_t *cq)
{
    struct mlx5_cqe64 *cqe;
    unsigned index;
    uint8_t op_own;

    index  = cq->cq_ci;
    cqe    = cq->cq_buf + (index & (cq->cq_length - 1)) * sizeof(struct mlx5_cqe64);
    op_own = cqe->op_own;

    if ((op_own & MLX5_CQE_OWNER_MASK) == !(index & cq->cq_length)) {
        return NULL;
    } else if (op_own & 0x80) {
        uct_ib_mlx5_check_completion(cqe);
        return NULL;
    }

    cq->cq_ci = index + 1;
    return cqe; /* TODO optimize - let complier know cqe is not null */
}


#endif
