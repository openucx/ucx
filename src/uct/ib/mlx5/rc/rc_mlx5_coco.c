/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_mlx5.h"
#include "rc_mlx5_coco.h"

#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/log.h>

#include <arpa/inet.h>
#include <string.h>


static uct_rc_mlx5_coco_qp_record_t *
uct_rc_mlx5_coco_qp_record_find(uct_rc_mlx5_coco_state_t *state,
                                uint32_t qpn)
{
    khiter_t iter;

    if (state == NULL) {
        return NULL;
    }

    iter = kh_get(uct_rc_mlx5_coco_qp_hash, &state->qp_hash, qpn);
    if (iter == kh_end(&state->qp_hash)) {
        return NULL;
    }

    return &state->qp_records[kh_value(&state->qp_hash, iter)];
}

static ucs_status_t
uct_rc_mlx5_coco_qp_record_index_add(uct_rc_mlx5_coco_state_t *state,
                                     uint32_t qpn, size_t index)
{
    khiter_t iter;
    int khret;

    iter = kh_put(uct_rc_mlx5_coco_qp_hash, &state->qp_hash, qpn, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        return UCS_ERR_NO_MEMORY;
    }

    if (khret == UCS_KH_PUT_KEY_PRESENT) {
        return UCS_ERR_ALREADY_EXISTS;
    }

    kh_value(&state->qp_hash, iter) = index;
    return UCS_OK;
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
    khiter_t iter;

    if (record == NULL) {
        return NULL;
    }

    iter = kh_get(uct_rc_mlx5_coco_tx_slot_hash, &record->tx_slot_hash,
                  wqe_counter);
    if (iter == kh_end(&record->tx_slot_hash)) {
        return NULL;
    }

    return &record->tx_slots[kh_value(&record->tx_slot_hash, iter)];
}

static ucs_status_t
uct_rc_mlx5_coco_tx_slot_index_add(uct_rc_mlx5_coco_qp_record_t *record,
                                   uint16_t wqe_counter, size_t index)
{
    khiter_t iter;
    int khret;

    iter = kh_put(uct_rc_mlx5_coco_tx_slot_hash, &record->tx_slot_hash,
                  wqe_counter, &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        return UCS_ERR_NO_MEMORY;
    }

    if (khret == UCS_KH_PUT_KEY_PRESENT) {
        return UCS_ERR_ALREADY_EXISTS;
    }

    kh_value(&record->tx_slot_hash, iter) = index;
    return UCS_OK;
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

    if (record->owner != NULL) {
        kh_destroy_inplace(uct_rc_mlx5_coco_tx_slot_hash,
                           &record->tx_slot_hash);
        ucs_free(record->tx_slots);
    }

    memset(record, 0, sizeof(*record));
    kh_init_inplace(uct_rc_mlx5_coco_tx_slot_hash, &record->tx_slot_hash);
    record->qpn        = qpn;
    record->generation = generation;
    record->state      = UCT_RC_MLX5_COCO_QP_LIVE;
    record->iface      = iface;
    record->tx_cq      = tx_cq;
    record->rx_cq      = rx_cq;
    record->ep         = ep;
    record->owner      = owner;
}

void uct_rc_mlx5_coco_state_init(uct_rc_mlx5_coco_state_t *state, int enabled)
{
    memset(state, 0, sizeof(*state));
    kh_init_inplace(uct_rc_mlx5_coco_qp_hash, &state->qp_hash);
    state->enabled = !!enabled;
}

void uct_rc_mlx5_coco_state_cleanup(uct_rc_mlx5_coco_state_t *state)
{
    size_t i;

    if (state == NULL) {
        return;
    }

    for (i = 0; i < state->qp_count; ++i) {
        kh_destroy_inplace(uct_rc_mlx5_coco_tx_slot_hash,
                           &state->qp_records[i].tx_slot_hash);
        ucs_free(state->qp_records[i].tx_slots);
    }

    kh_destroy_inplace(uct_rc_mlx5_coco_qp_hash, &state->qp_hash);
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

    record = &state->qp_records[state->qp_count];
    uct_rc_mlx5_coco_qp_record_reset(record, state, qpn, iface, tx_cq, rx_cq,
                                     ep);
    status = uct_rc_mlx5_coco_qp_record_index_add(state, qpn,
                                                  state->qp_count);
    if (status != UCS_OK) {
        kh_destroy_inplace(uct_rc_mlx5_coco_tx_slot_hash,
                           &record->tx_slot_hash);
        ucs_free(record->tx_slots);
        memset(record, 0, sizeof(*record));
        return status;
    }

    ++state->qp_count;
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

        slot = &record->tx_slots[record->tx_slot_count];
        memset(slot, 0, sizeof(*slot));
        slot->generation = 1;
        status = uct_rc_mlx5_coco_tx_slot_index_add(record, wqe_counter,
                                                    record->tx_slot_count);
        if (status != UCS_OK) {
            memset(slot, 0, sizeof(*slot));
            return status;
        }

        ++record->tx_slot_count;
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

    slot = uct_rc_mlx5_coco_tx_slot_find(record, hw_ci);
    if ((slot == NULL) || slot->retired) {
        return UCS_ERR_NO_ELEM;
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
