/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_RC_MLX5_COCO_H
#define UCT_RC_MLX5_COCO_H

#include <uct/ib/base/ib_iface.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/rc/base/rc_def.h>
#include <ucs/type/status.h>


typedef struct uct_rc_mlx5_iface_common uct_rc_mlx5_iface_common_t;
typedef struct uct_rc_mlx5_base_ep uct_rc_mlx5_base_ep_t;
typedef struct uct_rc_mlx5_coco_state uct_rc_mlx5_coco_state_t;

/*
 * CoCo RC mlx5 registry preconditions:
 * - Registry lookup is lock-free in the poll path because UCX worker progress
 *   serializes CQ polling and endpoint teardown for the owning worker.
 * - Registry mutation happens under the same worker/progress serialization, or
 *   before a QP/object is visible to hardware or the base QP table.
 * - Async callbacks must not free records visible to the poller without first
 *   poisoning and draining the affected object.
 * - Any future path that violates these preconditions must add synchronization
 *   outside the per-CQE hot path.
 */

typedef enum {
    UCT_RC_MLX5_COCO_QP_LIVE,
    UCT_RC_MLX5_COCO_QP_POISONED,
    UCT_RC_MLX5_COCO_QP_DRAINING,
    UCT_RC_MLX5_COCO_QP_DESTROYED
} uct_rc_mlx5_coco_qp_state_t;

enum {
    UCT_RC_MLX5_COCO_POISON_TX_CQ   = UCS_BIT(0),
    UCT_RC_MLX5_COCO_POISON_RX_CQ   = UCS_BIT(1),
    UCT_RC_MLX5_COCO_POISON_QP      = UCS_BIT(2),
    UCT_RC_MLX5_COCO_POISON_IFACE_TX = UCS_BIT(3),
    UCT_RC_MLX5_COCO_POISON_IFACE_RX = UCS_BIT(4)
};

typedef enum {
    UCT_RC_MLX5_COCO_TX_AM,
    UCT_RC_MLX5_COCO_TX_PUT,
    UCT_RC_MLX5_COCO_TX_GET,
    UCT_RC_MLX5_COCO_TX_FLUSH,
    UCT_RC_MLX5_COCO_TX_FC,
    UCT_RC_MLX5_COCO_TX_LAST
} uct_rc_mlx5_coco_tx_op_t;

typedef enum {
    UCT_RC_MLX5_COCO_SRQ_FREE,
    UCT_RC_MLX5_COCO_SRQ_POSTED,
    UCT_RC_MLX5_COCO_SRQ_COMPLETING,
    UCT_RC_MLX5_COCO_SRQ_CONSUMED,
    UCT_RC_MLX5_COCO_SRQ_POISONED
} uct_rc_mlx5_coco_srq_state_t;

typedef struct uct_rc_mlx5_coco_tx_slot {
    uint16_t                    wqe_counter;
    uint16_t                    generation;
    uct_rc_mlx5_coco_tx_op_t    op;
    uint8_t                     completion_expected;
    uint8_t                     retired;
    size_t                      expected_length;
    uct_rc_iface_send_op_t      *send_op;
} uct_rc_mlx5_coco_tx_slot_t;

typedef struct uct_rc_mlx5_coco_srq_slot {
    uct_rc_mlx5_coco_srq_state_t state;
    uint16_t                     generation;
    uint16_t                     slot;
    size_t                       posted_length;
    uct_ib_iface_recv_desc_t     *desc;
} uct_rc_mlx5_coco_srq_slot_t;

typedef struct uct_rc_mlx5_coco_qp_record {
    uint32_t                         qpn;
    uint32_t                         generation;
    uct_rc_mlx5_coco_qp_state_t      state;
    uct_rc_mlx5_iface_common_t       *iface;
    uct_ib_mlx5_cq_t                 *tx_cq;
    uct_ib_mlx5_cq_t                 *rx_cq;
    uct_rc_mlx5_base_ep_t            *ep;
    unsigned                         poison_scope;
    const char                       *poison_reason;
    uct_rc_mlx5_coco_state_t         *owner;
    uct_rc_mlx5_coco_tx_slot_t       *tx_slots;
    size_t                           tx_slot_count;
    size_t                           tx_slot_capacity;
    uint32_t                         srq_generation;
    uint8_t                          srq_attached;
} uct_rc_mlx5_coco_qp_record_t;

struct uct_rc_mlx5_coco_state {
    uint8_t                          enabled;
    uct_rc_mlx5_coco_qp_record_t     *qp_records;
    size_t                           qp_count;
    size_t                           qp_capacity;
    unsigned                         poison_scope;
    const char                       *poison_reason;
    uct_rc_mlx5_iface_common_t       *srq_iface;
    uct_ib_mlx5_cq_t                 *srq_rx_cq;
    uct_rc_mlx5_coco_srq_slot_t      *srq_slots;
    size_t                           srq_slot_count;
    uint32_t                         srq_generation;
};

typedef struct uct_rc_mlx5_coco_tx_cqe_result {
    uct_rc_mlx5_coco_qp_record_t *qp_record;
    uct_rc_mlx5_coco_tx_slot_t   *slot;
    uint16_t                     hw_ci;
} uct_rc_mlx5_coco_tx_cqe_result_t;

typedef struct uct_rc_mlx5_coco_rx_cqe_result {
    uct_rc_mlx5_coco_qp_record_t  *qp_record;
    uct_rc_mlx5_coco_srq_slot_t   *slot;
    uct_ib_iface_recv_desc_t      *desc;
    size_t                        length;
    uint32_t                      imm_data;
    uint8_t                       am_id;
} uct_rc_mlx5_coco_rx_cqe_result_t;

typedef struct uct_rc_mlx5_coco_error_cqe_result {
    uct_rc_mlx5_coco_qp_record_t  *qp_record;
    uct_rc_mlx5_coco_tx_slot_t    *tx_slot;
    uct_rc_mlx5_coco_srq_slot_t   *srq_slot;
    uint16_t                      wqe_counter;
    uct_ib_dir_t                  dir;
} uct_rc_mlx5_coco_error_cqe_result_t;

void uct_rc_mlx5_coco_state_init(uct_rc_mlx5_coco_state_t *state, int enabled);
void uct_rc_mlx5_coco_state_cleanup(uct_rc_mlx5_coco_state_t *state);

ucs_status_t
uct_rc_mlx5_coco_qp_record_add(uct_rc_mlx5_coco_state_t *state, uint32_t qpn,
                               uct_rc_mlx5_iface_common_t *iface,
                               uct_ib_mlx5_cq_t *tx_cq,
                               uct_ib_mlx5_cq_t *rx_cq,
                               uct_rc_mlx5_base_ep_t *ep,
                               uct_rc_mlx5_coco_qp_record_t **record_p);

ucs_status_t
uct_rc_mlx5_coco_qp_record_destroy(uct_rc_mlx5_coco_state_t *state,
                                   uint32_t qpn);

ucs_status_t
uct_rc_mlx5_coco_qp_record_validate(uct_rc_mlx5_coco_state_t *state,
                                    uint32_t qpn, uint32_t generation,
                                    uct_rc_mlx5_iface_common_t *iface,
                                    uct_ib_mlx5_cq_t *cq,
                                    uct_ib_dir_t dir,
                                    uct_rc_mlx5_coco_qp_record_t **record_p);

uct_rc_mlx5_coco_qp_record_t *
uct_rc_mlx5_coco_qp_record_lookup(uct_rc_mlx5_coco_state_t *state,
                                  uint32_t qpn);

ucs_status_t
uct_rc_mlx5_coco_poison(uct_rc_mlx5_iface_common_t *iface,
                        uct_rc_mlx5_coco_qp_record_t *qp_record,
                        uct_ib_mlx5_cq_t *cq, unsigned poison_scope,
                        const char *reason);

ucs_status_t
uct_rc_mlx5_coco_tx_shadow_record(uct_rc_mlx5_coco_qp_record_t *record,
                                  uint16_t wqe_counter,
                                  uct_rc_mlx5_coco_tx_op_t op,
                                  int completion_expected,
                                  size_t expected_length,
                                  uct_rc_iface_send_op_t *send_op,
                                  uct_rc_mlx5_coco_tx_slot_t **slot_p);

ucs_status_t
uct_rc_mlx5_coco_tx_shadow_validate(uct_rc_mlx5_coco_qp_record_t *record,
                                    uint16_t wqe_counter,
                                    uint16_t generation,
                                    uct_rc_mlx5_coco_tx_slot_t **slot_p);

ucs_status_t
uct_rc_mlx5_coco_tx_shadow_retire(uct_rc_mlx5_coco_qp_record_t *record,
                                  uint16_t wqe_counter, uint16_t generation,
                                  uct_rc_mlx5_coco_tx_slot_t **slot_p);

ucs_status_t
uct_rc_mlx5_coco_tx_cqe_validate(uct_rc_mlx5_coco_state_t *state,
                                 uct_ib_mlx5_cq_t *cq,
                                 const struct mlx5_cqe64 *cqe,
                                 uct_rc_mlx5_coco_tx_cqe_result_t *result);

ucs_status_t
uct_rc_mlx5_coco_rx_cqe_validate(uct_rc_mlx5_coco_state_t *state,
                                 uct_ib_mlx5_cq_t *cq,
                                 const struct mlx5_cqe64 *cqe,
                                 uct_rc_mlx5_coco_rx_cqe_result_t *result);

ucs_status_t
uct_rc_mlx5_coco_error_cqe_validate(
        uct_rc_mlx5_coco_state_t *state, uct_ib_mlx5_cq_t *cq,
        uct_ib_dir_t dir, const struct mlx5_cqe64 *cqe,
        uct_rc_mlx5_coco_error_cqe_result_t *result);

ucs_status_t
uct_rc_mlx5_coco_error_cqe_poison(
        uct_rc_mlx5_iface_common_t *iface, uct_ib_mlx5_cq_t *cq,
        uct_ib_dir_t dir, uct_rc_mlx5_coco_error_cqe_result_t *result,
        const char *reason);

uct_rc_mlx5_coco_tx_op_t
uct_rc_mlx5_coco_tx_op_from_opcode(uint8_t opcode);

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_init(uct_rc_mlx5_coco_state_t *state,
                                 uct_rc_mlx5_iface_common_t *iface,
                                 uct_ib_mlx5_cq_t *rx_cq,
                                 size_t slot_count);

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_post(uct_rc_mlx5_coco_state_t *state,
                                 uint16_t slot_index, size_t posted_length,
                                 uct_ib_iface_recv_desc_t *desc,
                                 uct_rc_mlx5_coco_srq_slot_t **slot_p);

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_validate(uct_rc_mlx5_coco_state_t *state,
                                     uint32_t qpn, uct_ib_mlx5_cq_t *rx_cq,
                                     uint16_t slot_index, uint16_t generation,
                                     size_t byte_count,
                                     uct_rc_mlx5_coco_srq_slot_t **slot_p);

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_consume(uct_rc_mlx5_coco_state_t *state,
                                    uint16_t slot_index, uint16_t generation);

#endif
