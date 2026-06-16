/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_mlx5_common.h"
#include "rc_mlx5_coco.h"

#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/log.h>

#include <arpa/inet.h>
#include <string.h>


static uct_rc_mlx5_coco_qp_record_t *
uct_rc_mlx5_coco_qp_record_find(uct_rc_mlx5_coco_state_t *state,
                                uint32_t qpn)
{
    size_t i;

    if (state == NULL) {
        return NULL;
    }

    for (i = 0; i < state->qp_count; ++i) {
        if (state->qp_records[i].qpn == qpn) {
            return &state->qp_records[i];
        }
    }

    return NULL;
}

static ucs_status_t
uct_rc_mlx5_coco_qp_records_reserve(uct_rc_mlx5_coco_state_t *state)
{
    uct_rc_mlx5_coco_qp_record_t *records;
    size_t new_capacity;

    if (state->qp_count < state->qp_capacity) {
        return UCS_OK;
    }

    new_capacity = (state->qp_capacity == 0) ? 8 : state->qp_capacity * 2;
    records      = ucs_realloc(state->qp_records,
                               new_capacity * sizeof(*records),
                               "rc mlx5 coco qp records");
    if (records == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    memset(records + state->qp_capacity, 0,
           (new_capacity - state->qp_capacity) * sizeof(*records));
    state->qp_records   = records;
    state->qp_capacity  = new_capacity;
    return UCS_OK;
}

static uct_rc_mlx5_coco_tx_slot_t *
uct_rc_mlx5_coco_tx_slot_find(uct_rc_mlx5_coco_qp_record_t *record,
                              uint16_t wqe_counter)
{
    size_t i;

    if (record == NULL) {
        return NULL;
    }

    for (i = 0; i < record->tx_slot_count; ++i) {
        if (record->tx_slots[i].wqe_counter == wqe_counter) {
            return &record->tx_slots[i];
        }
    }

    return NULL;
}

static ucs_status_t
uct_rc_mlx5_coco_tx_slots_reserve(uct_rc_mlx5_coco_qp_record_t *record)
{
    uct_rc_mlx5_coco_tx_slot_t *slots;
    size_t new_capacity;

    if (record->tx_slot_count < record->tx_slot_capacity) {
        return UCS_OK;
    }

    new_capacity = (record->tx_slot_capacity == 0) ? 16 :
                   record->tx_slot_capacity * 2;
    slots        = ucs_realloc(record->tx_slots,
                               new_capacity * sizeof(*slots),
                               "rc mlx5 coco tx slots");
    if (slots == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    memset(slots + record->tx_slot_capacity, 0,
           (new_capacity - record->tx_slot_capacity) * sizeof(*slots));
    record->tx_slots         = slots;
    record->tx_slot_capacity = new_capacity;
    return UCS_OK;
}

static uint16_t uct_rc_mlx5_coco_next_u16_generation(uint16_t generation)
{
    ++generation;
    return (generation == 0) ? 1 : generation;
}

static uct_rc_mlx5_coco_srq_slot_t *
uct_rc_mlx5_coco_srq_slot_get(uct_rc_mlx5_coco_state_t *state,
                              uint16_t slot_index)
{
    if ((state == NULL) || (state->srq_slots == NULL) ||
        (state->srq_slot_count == 0)) {
        return NULL;
    }

    return &state->srq_slots[slot_index % state->srq_slot_count];
}

static void
uct_rc_mlx5_coco_tx_shadow_retire_to(uct_rc_mlx5_coco_qp_record_t *record,
                                     uint16_t hw_ci, uint16_t generation)
{
    uct_rc_mlx5_coco_tx_slot_t *slot;
    size_t i;

    for (i = 0; i < record->tx_slot_count; ++i) {
        slot = &record->tx_slots[i];
        if (!slot->retired && (slot->generation == generation) &&
            UCS_CIRCULAR_COMPARE16(slot->wqe_counter, <=, hw_ci)) {
            slot->retired = 1;
        }
    }
}

static void
uct_rc_mlx5_coco_qp_record_reset(uct_rc_mlx5_coco_qp_record_t *record,
                                 uct_rc_mlx5_coco_state_t *owner,
                                 uint32_t qpn,
                                 uct_rc_mlx5_iface_common_t *iface,
                                 uct_ib_mlx5_cq_t *tx_cq,
                                 uct_ib_mlx5_cq_t *rx_cq,
                                 uct_rc_mlx5_base_ep_t *ep)
{
    uint32_t generation = record->generation + 1;

    if (generation == 0) {
        generation = 1;
    }

    ucs_free(record->tx_slots);
    memset(record, 0, sizeof(*record));
    record->qpn        = qpn;
    record->generation = generation;
    record->state      = UCT_RC_MLX5_COCO_QP_LIVE;
    record->iface      = iface;
    record->tx_cq      = tx_cq;
    record->rx_cq      = rx_cq;
    record->ep         = ep;
    record->owner      = owner;

    if (owner->srq_slots != NULL) {
        record->srq_attached    = 1;
        record->srq_generation  = owner->srq_generation;
    }
}

void uct_rc_mlx5_coco_state_init(uct_rc_mlx5_coco_state_t *state, int enabled)
{
    memset(state, 0, sizeof(*state));
    state->enabled = !!enabled;
}

void uct_rc_mlx5_coco_state_cleanup(uct_rc_mlx5_coco_state_t *state)
{
    size_t i;

    if (state == NULL) {
        return;
    }

    for (i = 0; i < state->qp_count; ++i) {
        ucs_free(state->qp_records[i].tx_slots);
    }

    ucs_free(state->srq_slots);
    ucs_free(state->qp_records);
    memset(state, 0, sizeof(*state));
}

ucs_status_t
uct_rc_mlx5_coco_qp_record_add(uct_rc_mlx5_coco_state_t *state, uint32_t qpn,
                               uct_rc_mlx5_iface_common_t *iface,
                               uct_ib_mlx5_cq_t *tx_cq,
                               uct_ib_mlx5_cq_t *rx_cq,
                               uct_rc_mlx5_base_ep_t *ep,
                               uct_rc_mlx5_coco_qp_record_t **record_p)
{
    uct_rc_mlx5_coco_qp_record_t *record;
    ucs_status_t status;

    if (record_p != NULL) {
        *record_p = NULL;
    }

    if ((state == NULL) || (qpn == 0) || (tx_cq == NULL) || (rx_cq == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
    if (record != NULL) {
        if (record->state != UCT_RC_MLX5_COCO_QP_DESTROYED) {
            return UCS_ERR_ALREADY_EXISTS;
        }

        uct_rc_mlx5_coco_qp_record_reset(record, state, qpn, iface, tx_cq,
                                         rx_cq, ep);
        if (record_p != NULL) {
            *record_p = record;
        }
        return UCS_OK;
    }

    status = uct_rc_mlx5_coco_qp_records_reserve(state);
    if (status != UCS_OK) {
        return status;
    }

    record = &state->qp_records[state->qp_count++];
    uct_rc_mlx5_coco_qp_record_reset(record, state, qpn, iface, tx_cq, rx_cq,
                                     ep);
    if (record_p != NULL) {
        *record_p = record;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_qp_record_destroy(uct_rc_mlx5_coco_state_t *state,
                                   uint32_t qpn)
{
    uct_rc_mlx5_coco_qp_record_t *record;

    record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
    if (record == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    record->state = UCT_RC_MLX5_COCO_QP_DESTROYED;
    return UCS_OK;
}

uct_rc_mlx5_coco_qp_record_t *
uct_rc_mlx5_coco_qp_record_lookup(uct_rc_mlx5_coco_state_t *state,
                                  uint32_t qpn)
{
    return uct_rc_mlx5_coco_qp_record_find(state, qpn);
}

ucs_status_t
uct_rc_mlx5_coco_qp_record_validate(uct_rc_mlx5_coco_state_t *state,
                                    uint32_t qpn, uint32_t generation,
                                    uct_rc_mlx5_iface_common_t *iface,
                                    uct_ib_mlx5_cq_t *cq,
                                    uct_ib_dir_t dir,
                                    uct_rc_mlx5_coco_qp_record_t **record_p)
{
    uct_rc_mlx5_coco_qp_record_t *record;
    uct_ib_mlx5_cq_t *expected_cq;

    if (record_p != NULL) {
        *record_p = NULL;
    }

    record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
    if (record == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    expected_cq = (dir == UCT_IB_DIR_TX) ? record->tx_cq : record->rx_cq;
    if ((record->state != UCT_RC_MLX5_COCO_QP_LIVE) ||
        (record->generation != generation) || (record->iface != iface) ||
        (expected_cq != cq)) {
        return UCS_ERR_IO_ERROR;
    }

    if (record_p != NULL) {
        *record_p = record;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_poison(uct_rc_mlx5_iface_common_t *iface,
                        uct_rc_mlx5_coco_qp_record_t *qp_record,
                        uct_ib_mlx5_cq_t *cq, unsigned poison_scope,
                        const char *reason)
{
    uct_rc_mlx5_coco_state_t *state = NULL;

    if (iface != NULL) {
        state = &iface->coco;
    } else if (qp_record != NULL) {
        state = qp_record->owner;
    }

    if (state != NULL) {
        state->poison_scope |= poison_scope;
        state->poison_reason = reason;
    }

    if (qp_record != NULL) {
        if (poison_scope & UCT_RC_MLX5_COCO_POISON_QP) {
            qp_record->state = UCT_RC_MLX5_COCO_QP_POISONED;
        }
        qp_record->poison_scope |= poison_scope;
        qp_record->poison_reason = reason;
    }

    ucs_debug("rc mlx5 CoCo poison iface %p qp_record %p cq %p scope 0x%x: %s",
              iface, qp_record, cq, poison_scope, reason);
    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_tx_shadow_record(uct_rc_mlx5_coco_qp_record_t *record,
                                  uint16_t wqe_counter,
                                  uct_rc_mlx5_coco_tx_op_t op,
                                  int completion_expected,
                                  size_t expected_length,
                                  uct_rc_iface_send_op_t *send_op,
                                  uct_rc_mlx5_coco_tx_slot_t **slot_p)
{
    uct_rc_mlx5_coco_tx_slot_t *slot;
    ucs_status_t status;
    uint16_t generation;

    if (slot_p != NULL) {
        *slot_p = NULL;
    }

    if ((record == NULL) || (record->state != UCT_RC_MLX5_COCO_QP_LIVE) ||
        (op >= UCT_RC_MLX5_COCO_TX_LAST)) {
        return UCS_ERR_INVALID_PARAM;
    }

    slot = uct_rc_mlx5_coco_tx_slot_find(record, wqe_counter);
    if (slot != NULL) {
        if (!slot->retired) {
            return UCS_ERR_ALREADY_EXISTS;
        }

        generation = slot->generation + 1;
        if (generation == 0) {
            generation = 1;
        }

        memset(slot, 0, sizeof(*slot));
        slot->generation = generation;
    } else {
        status = uct_rc_mlx5_coco_tx_slots_reserve(record);
        if (status != UCS_OK) {
            return status;
        }

        slot = &record->tx_slots[record->tx_slot_count++];
        memset(slot, 0, sizeof(*slot));
        slot->generation = 1;
    }

    slot->wqe_counter        = wqe_counter;
    slot->op                 = op;
    slot->completion_expected = !!completion_expected;
    slot->expected_length    = expected_length;
    slot->send_op            = send_op;

    if (slot_p != NULL) {
        *slot_p = slot;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_tx_shadow_validate(uct_rc_mlx5_coco_qp_record_t *record,
                                    uint16_t wqe_counter,
                                    uint16_t generation,
                                    uct_rc_mlx5_coco_tx_slot_t **slot_p)
{
    uct_rc_mlx5_coco_tx_slot_t *slot;

    if (slot_p != NULL) {
        *slot_p = NULL;
    }

    slot = uct_rc_mlx5_coco_tx_slot_find(record, wqe_counter);
    if ((slot == NULL) || slot->retired) {
        return UCS_ERR_NO_ELEM;
    }

    if (slot->generation != generation) {
        return UCS_ERR_IO_ERROR;
    }

    if (slot_p != NULL) {
        *slot_p = slot;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_tx_shadow_retire(uct_rc_mlx5_coco_qp_record_t *record,
                                  uint16_t wqe_counter, uint16_t generation,
                                  uct_rc_mlx5_coco_tx_slot_t **slot_p)
{
    uct_rc_mlx5_coco_tx_slot_t *slot;
    ucs_status_t status;

    status = uct_rc_mlx5_coco_tx_shadow_validate(record, wqe_counter,
                                                 generation, &slot);
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_mlx5_coco_tx_shadow_retire_to(record, wqe_counter,
                                         slot->generation);
    if (slot_p != NULL) {
        *slot_p = slot;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_tx_cqe_validate(uct_rc_mlx5_coco_state_t *state,
                                 uct_ib_mlx5_cq_t *cq,
                                 const struct mlx5_cqe64 *cqe,
                                 uct_rc_mlx5_coco_tx_cqe_result_t *result)
{
    uct_rc_mlx5_coco_qp_record_t *record = NULL;
    uct_rc_mlx5_coco_tx_slot_t *slot;
    uint32_t qpn;
    uint16_t hw_ci;
    uint8_t opcode;
    ucs_status_t status;

    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }

    if ((state == NULL) || (cq == NULL) || (cqe == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    opcode = cqe->op_own >> 4;
    if (opcode != MLX5_CQE_REQ) {
        return UCS_ERR_IO_ERROR;
    }

    qpn    = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    hw_ci  = ntohs(cqe->wqe_counter);
    record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
    if (record == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    status = uct_rc_mlx5_coco_qp_record_validate(state, qpn,
                                                 record->generation,
                                                 record->iface, cq,
                                                 UCT_IB_DIR_TX, &record);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rc_mlx5_coco_tx_shadow_validate(record, hw_ci,
                                                 uct_rc_mlx5_coco_tx_slot_find(
                                                         record, hw_ci) ?
                                                 uct_rc_mlx5_coco_tx_slot_find(
                                                         record, hw_ci)->generation :
                                                 0, &slot);
    if (status != UCS_OK) {
        return status;
    }

    if (!slot->completion_expected) {
        return UCS_ERR_IO_ERROR;
    }

    uct_rc_mlx5_coco_tx_shadow_retire_to(record, hw_ci, slot->generation);
    if (result != NULL) {
        result->qp_record = record;
        result->slot      = slot;
        result->hw_ci     = hw_ci;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_rx_cqe_validate(uct_rc_mlx5_coco_state_t *state,
                                 uct_ib_mlx5_cq_t *cq,
                                 const struct mlx5_cqe64 *cqe,
                                 uct_rc_mlx5_coco_rx_cqe_result_t *result)
{
    uct_rc_mlx5_coco_qp_record_t *record = NULL;
    uct_rc_mlx5_coco_srq_slot_t *slot;
    uint16_t wqe_counter;
    uint32_t qpn;
    uint8_t opcode;
    size_t byte_count;
    ucs_status_t status;

    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }

    if ((state == NULL) || (cq == NULL) || (cqe == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    if ((cqe->op_own & UCT_IB_MLX5_CQE_FORMAT_MASK) ==
        UCT_IB_MLX5_CQE_FORMAT_MASK) {
        return UCS_ERR_IO_ERROR;
    }

    if (cqe->op_own & (MLX5_INLINE_SCATTER_32 | MLX5_INLINE_SCATTER_64)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (cqe->app == UCT_RC_MLX5_CQE_APP_TAG_MATCHING) {
        return UCS_ERR_UNSUPPORTED;
    }

    opcode = cqe->op_own >> 4;
    if ((opcode != MLX5_CQE_RESP_SEND) &&
        (opcode != MLX5_CQE_RESP_SEND_IMM)) {
        return UCS_ERR_IO_ERROR;
    }

    qpn         = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    wqe_counter = ntohs(cqe->wqe_counter);
    byte_count  = ntohl(cqe->byte_cnt) & UCT_IB_MLX5_MP_RQ_BYTE_CNT_MASK;
    slot        = uct_rc_mlx5_coco_srq_slot_get(state, wqe_counter);
    if (slot == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    status = uct_rc_mlx5_coco_srq_shadow_validate(
            state, qpn, cq, wqe_counter, slot->generation, byte_count, &slot);
    if (status != UCS_OK) {
        return status;
    }

    record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
    ucs_assert(record != NULL);

    if (result != NULL) {
        result->qp_record = record;
        result->slot      = slot;
        result->desc      = slot->desc;
        result->length    = byte_count;
        result->imm_data  = cqe->imm_inval_pkey;
        result->am_id     = 0;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_error_cqe_validate(
        uct_rc_mlx5_coco_state_t *state, uct_ib_mlx5_cq_t *cq,
        uct_ib_dir_t dir, const struct mlx5_cqe64 *cqe,
        uct_rc_mlx5_coco_error_cqe_result_t *result)
{
    uct_rc_mlx5_coco_qp_record_t *record = NULL;
    uct_rc_mlx5_coco_srq_slot_t *srq_slot;
    uct_rc_mlx5_coco_tx_slot_t *tx_slot;
    const uct_ib_mlx5_err_cqe_t *ecqe = (const void*)cqe;
    uint16_t wqe_counter;
    uint32_t qpn;
    uint8_t opcode;
    ucs_status_t status;

    if (result != NULL) {
        memset(result, 0, sizeof(*result));
        result->dir = dir;
    }

    if ((state == NULL) || (cq == NULL) || (cqe == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    opcode      = ecqe->op_own >> 4;
    qpn         = ntohl(ecqe->s_wqe_opcode_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    wqe_counter = ntohs(ecqe->wqe_counter);

    if (dir == UCT_IB_DIR_RX) {
        if (opcode != MLX5_CQE_RESP_ERR) {
            return UCS_ERR_IO_ERROR;
        }

        srq_slot = uct_rc_mlx5_coco_srq_slot_get(state, wqe_counter);
        if (srq_slot == NULL) {
            return UCS_ERR_NO_ELEM;
        }

        status = uct_rc_mlx5_coco_srq_shadow_validate(
                state, qpn, cq, wqe_counter, srq_slot->generation, 0,
                &srq_slot);
        if (status != UCS_OK) {
            return status;
        }

        record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
        ucs_assert(record != NULL);

        if (result != NULL) {
            result->qp_record   = record;
            result->srq_slot    = srq_slot;
            result->wqe_counter = wqe_counter;
        }

        return UCS_OK;
    }

    if (dir == UCT_IB_DIR_TX) {
        if (opcode != MLX5_CQE_REQ_ERR) {
            return UCS_ERR_IO_ERROR;
        }

        record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
        if (record == NULL) {
            return UCS_ERR_NO_ELEM;
        }

        status = uct_rc_mlx5_coco_qp_record_validate(state, qpn,
                                                     record->generation,
                                                     record->iface, cq,
                                                     UCT_IB_DIR_TX, &record);
        if (status != UCS_OK) {
            return status;
        }

        tx_slot = uct_rc_mlx5_coco_tx_slot_find(record, wqe_counter);
        if (tx_slot == NULL) {
            return UCS_ERR_NO_ELEM;
        }

        status = uct_rc_mlx5_coco_tx_shadow_validate(record, wqe_counter,
                                                     tx_slot->generation,
                                                     &tx_slot);
        if (status != UCS_OK) {
            return status;
        }

        uct_rc_mlx5_coco_tx_shadow_retire_to(record, wqe_counter,
                                             tx_slot->generation);
        if (result != NULL) {
            result->qp_record   = record;
            result->tx_slot     = tx_slot;
            result->wqe_counter = wqe_counter;
        }

        return UCS_OK;
    }

    return UCS_ERR_INVALID_PARAM;
}

ucs_status_t
uct_rc_mlx5_coco_error_cqe_poison(
        uct_rc_mlx5_iface_common_t *iface, uct_ib_mlx5_cq_t *cq,
        uct_ib_dir_t dir, uct_rc_mlx5_coco_error_cqe_result_t *result,
        const char *reason)
{
    uct_rc_mlx5_coco_qp_record_t *record = NULL;
    unsigned poison_scope;

    poison_scope = (dir == UCT_IB_DIR_TX) ?
                   (UCT_RC_MLX5_COCO_POISON_TX_CQ |
                    UCT_RC_MLX5_COCO_POISON_IFACE_TX) :
                   (UCT_RC_MLX5_COCO_POISON_RX_CQ |
                    UCT_RC_MLX5_COCO_POISON_IFACE_RX);
    if (result != NULL) {
        record = result->qp_record;
    }

    if (record != NULL) {
        poison_scope |= UCT_RC_MLX5_COCO_POISON_QP;
    }

    return uct_rc_mlx5_coco_poison(iface, record, cq, poison_scope, reason);
}

uct_rc_mlx5_coco_tx_op_t
uct_rc_mlx5_coco_tx_op_from_opcode(uint8_t opcode)
{
    switch (opcode) {
    case MLX5_OPCODE_SEND:
    case MLX5_OPCODE_SEND_IMM:
        return UCT_RC_MLX5_COCO_TX_AM;
    case MLX5_OPCODE_RDMA_WRITE:
        return UCT_RC_MLX5_COCO_TX_PUT;
    case MLX5_OPCODE_RDMA_READ:
        return UCT_RC_MLX5_COCO_TX_GET;
    case MLX5_OPCODE_NOP:
        return UCT_RC_MLX5_COCO_TX_FLUSH;
    default:
        return UCT_RC_MLX5_COCO_TX_LAST;
    }
}

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_init(uct_rc_mlx5_coco_state_t *state,
                                 uct_rc_mlx5_iface_common_t *iface,
                                 uct_ib_mlx5_cq_t *rx_cq, size_t slot_count)
{
    uct_rc_mlx5_coco_srq_slot_t *slots;
    uint32_t generation;
    size_t i;

    if ((state == NULL) || (iface == NULL) || (rx_cq == NULL) ||
        (slot_count == 0)) {
        return UCS_ERR_INVALID_PARAM;
    }

    slots = ucs_calloc(slot_count, sizeof(*slots), "rc mlx5 coco srq slots");
    if (slots == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    generation = state->srq_generation + 1;
    if (generation == 0) {
        generation = 1;
    }

    ucs_free(state->srq_slots);
    state->srq_iface      = iface;
    state->srq_rx_cq      = rx_cq;
    state->srq_slots      = slots;
    state->srq_slot_count = slot_count;
    state->srq_generation = generation;

    for (i = 0; i < slot_count; ++i) {
        state->srq_slots[i].state = UCT_RC_MLX5_COCO_SRQ_FREE;
        state->srq_slots[i].slot  = i;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_post(uct_rc_mlx5_coco_state_t *state,
                                 uint16_t slot_index, size_t posted_length,
                                 uct_ib_iface_recv_desc_t *desc,
                                 uct_rc_mlx5_coco_srq_slot_t **slot_p)
{
    uct_rc_mlx5_coco_srq_slot_t *slot;

    if (slot_p != NULL) {
        *slot_p = NULL;
    }

    if ((posted_length == 0) || (desc == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    slot = uct_rc_mlx5_coco_srq_slot_get(state, slot_index);
    if (slot == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    if ((slot->state == UCT_RC_MLX5_COCO_SRQ_POSTED) ||
        (slot->state == UCT_RC_MLX5_COCO_SRQ_COMPLETING) ||
        (slot->state == UCT_RC_MLX5_COCO_SRQ_POISONED)) {
        return UCS_ERR_ALREADY_EXISTS;
    }

    slot->generation    = uct_rc_mlx5_coco_next_u16_generation(slot->generation);
    slot->state         = UCT_RC_MLX5_COCO_SRQ_POSTED;
    slot->slot          = slot_index;
    slot->posted_length = posted_length;
    slot->desc          = desc;

    if (slot_p != NULL) {
        *slot_p = slot;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_validate(uct_rc_mlx5_coco_state_t *state,
                                     uint32_t qpn, uct_ib_mlx5_cq_t *rx_cq,
                                     uint16_t slot_index, uint16_t generation,
                                     size_t byte_count,
                                     uct_rc_mlx5_coco_srq_slot_t **slot_p)
{
    uct_rc_mlx5_coco_qp_record_t *record;
    uct_rc_mlx5_coco_srq_slot_t *slot;
    ucs_status_t status;

    if (slot_p != NULL) {
        *slot_p = NULL;
    }

    if ((state == NULL) || (rx_cq == NULL) || (rx_cq != state->srq_rx_cq)) {
        return UCS_ERR_INVALID_PARAM;
    }

    slot = uct_rc_mlx5_coco_srq_slot_get(state, slot_index);
    if ((slot == NULL) || (slot->state != UCT_RC_MLX5_COCO_SRQ_POSTED)) {
        return UCS_ERR_NO_ELEM;
    }

    if (slot->slot != slot_index) {
        return UCS_ERR_IO_ERROR;
    }

    if (slot->generation != generation) {
        return UCS_ERR_IO_ERROR;
    }

    if (byte_count > slot->posted_length) {
        return UCS_ERR_INVALID_PARAM;
    }

    record = uct_rc_mlx5_coco_qp_record_find(state, qpn);
    if (record == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    status = uct_rc_mlx5_coco_qp_record_validate(
            state, qpn, record->generation, state->srq_iface, rx_cq,
            UCT_IB_DIR_RX, &record);
    if (status != UCS_OK) {
        return status;
    }

    if (!record->srq_attached ||
        (record->srq_generation != state->srq_generation)) {
        return UCS_ERR_IO_ERROR;
    }

    slot->state = UCT_RC_MLX5_COCO_SRQ_COMPLETING;

    if (slot_p != NULL) {
        *slot_p = slot;
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_mlx5_coco_srq_shadow_consume(uct_rc_mlx5_coco_state_t *state,
                                    uint16_t slot_index, uint16_t generation)
{
    uct_rc_mlx5_coco_srq_slot_t *slot;

    slot = uct_rc_mlx5_coco_srq_slot_get(state, slot_index);
    if ((slot == NULL) ||
        ((slot->state != UCT_RC_MLX5_COCO_SRQ_POSTED) &&
         (slot->state != UCT_RC_MLX5_COCO_SRQ_COMPLETING))) {
        return UCS_ERR_NO_ELEM;
    }

    if (slot->slot != slot_index) {
        return UCS_ERR_IO_ERROR;
    }

    if (slot->generation != generation) {
        return UCS_ERR_IO_ERROR;
    }

    slot->state         = UCT_RC_MLX5_COCO_SRQ_CONSUMED;
    slot->posted_length = 0;
    slot->desc          = NULL;
    return UCS_OK;
}
