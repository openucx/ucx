/**
* Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_ECE_H_
#define UCT_ECE_H_


#include <uct/ib/base/ib_log.h>
#include <uct/ib/base/ib_device.h>
#include <ucs/debug/log.h>
#include <ucs/type/status.h>

#define UCT_IB_MLX5_VENDOR_ID             0x15b3
#define UCT_IB_MLX5_ECE_VER_MAX           2
#define UCT_IB_MLX5_ECE_VER_SR            2
#define UCT_IB_MLX5_ECE_VER_SHIFT         28
#define UCT_IB_MLX5_ECE_SR                0x1
#define UCT_IB_MLX5_ECE_UNKNOWN_MASK      0x0ffffffe


typedef struct {
    int selective_repeat;
} uct_ib_mlx5_ece_fields_t;


static inline uint32_t uct_ib_mlx5_ece(uct_ib_mlx5_qp_t *qp) {
    uct_ib_ece ece;

    if (qp->type == UCT_IB_MLX5_OBJ_TYPE_VERBS) {
        uct_ib_query_ece(qp->verbs.qp, &ece);
        if (ece.vendor_id == UCT_IB_MLX5_VENDOR_ID) {
            return ece.options;
        }
    } else if (qp->type == UCT_IB_MLX5_OBJ_TYPE_DEVX) {
#if HAVE_DEVX
        return qp->devx.ece;
#endif
    }

    return 0;
}

static inline void uct_ib_mlx5_set_ece(uint32_t ece, uct_ib_ece *ibece) {
    memset(ibece, 0, sizeof(*ibece));
    ibece->vendor_id = UCT_IB_MLX5_VENDOR_ID;
    ibece->options   = ece;
}

static inline int uct_ib_mlx5_decode_ece(uint32_t ece, uct_ib_mlx5_ece_fields_t *fields) {
    int version = ece >> UCT_IB_MLX5_ECE_VER_SHIFT;

    if ((version > UCT_IB_MLX5_ECE_VER_MAX) &&
        (ece & UCT_IB_MLX5_ECE_UNKNOWN_MASK)) {
        return UCS_ERR_UNSUPPORTED;
    }

    memset(fields, 0, sizeof(*fields));
    switch (version) {
        case UCT_IB_MLX5_ECE_VER_SR:
            fields->selective_repeat = !!(ece & UCT_IB_MLX5_ECE_SR);
    }

    return UCS_OK;
}

static inline uint32_t uct_ib_mlx5_encode_ece(uct_ib_mlx5_ece_fields_t *fields) {
    uint32_t ece = UCT_IB_MLX5_ECE_VER_MAX << UCT_IB_MLX5_ECE_VER_SHIFT;

    if (fields->selective_repeat) {
        ece |= UCT_IB_MLX5_ECE_SR;
    }

    return ece;
}

#endif
