/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_mlx5_log.h"

#include <uct/ib/base/ib_device.h>
#include <string.h>


static const char *uct_ib_mlx5_cqe_err_opcode(struct mlx5_err_cqe *ecqe)
{
    uint8_t wqe_err_opcode = ntohl(ecqe->s_wqe_opcode_qpn) >> 24;

    switch (ecqe->op_own >> 4) {
    case MLX5_CQE_REQ_ERR:
        switch (wqe_err_opcode) {
        case MLX5_OPCODE_RDMA_WRITE_IMM:
        case MLX5_OPCODE_RDMA_WRITE:
            return "RDMA_WRITE";
        case MLX5_OPCODE_SEND_IMM:
        case MLX5_OPCODE_SEND:
        case MLX5_OPCODE_SEND_INVAL:
            return "SEND";
        case MLX5_OPCODE_RDMA_READ:
            return "RDMA_READ";
        case MLX5_OPCODE_ATOMIC_CS:
            return "COMPARE_SWAP";
        case MLX5_OPCODE_ATOMIC_FA:
            return "FETCH_ADD";
        case MLX5_OPCODE_BIND_MW:
            return "BIND_MW";
        case MLX5_OPCODE_ATOMIC_MASKED_CS:
            return "MASKED_COMPARE_SWAP";
        case MLX5_OPCODE_ATOMIC_MASKED_FA:
            return "MASKED_FETCH_ADD";
        default:
            return "";
        }
    case MLX5_CQE_RESP_ERR:
        return "RECV";
    default:
        return "";
    }
}

void uct_ib_mlx5_completion_with_err(struct mlx5_err_cqe *ecqe)
{
    uint16_t wqe_counter;
    uint32_t qp_num;
    char info[200] = {0};

    wqe_counter = ntohs(ecqe->wqe_counter);
    qp_num      = ntohl(ecqe->s_wqe_opcode_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);

    switch (ecqe->syndrome) {
    case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR:
        snprintf(info, sizeof(info), "Local length");
        break;
    case MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR:
        snprintf(info, sizeof(info), "Local QP operation");
        break;
    case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR:
        snprintf(info, sizeof(info), "Local protection");
        break;
    case MLX5_CQE_SYNDROME_WR_FLUSH_ERR:
        snprintf(info, sizeof(info), "WR flushed because QP in error state");
        break;
    case MLX5_CQE_SYNDROME_MW_BIND_ERR:
        snprintf(info, sizeof(info), "Memory window bind");
        break;
    case MLX5_CQE_SYNDROME_BAD_RESP_ERR:
        snprintf(info, sizeof(info), "Bad response");
        break;
    case MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR:
        snprintf(info, sizeof(info), "Local access");
        break;
    case MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR:
        snprintf(info, sizeof(info), "Invalid request");
        break;
    case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR:
        snprintf(info, sizeof(info), "Remote access");
        break;
    case MLX5_CQE_SYNDROME_REMOTE_OP_ERR:
        snprintf(info, sizeof(info), "Remote QP");
        break;
    case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR:
        snprintf(info, sizeof(info), "Transport retry count exceeded");
        break;
    case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR:
        snprintf(info, sizeof(info), "Receive-no-ready retry count exceeded");
        break;
    case MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR:
        snprintf(info, sizeof(info), "Remote side aborted");
        break;
    default:
        snprintf(info, sizeof(info), "Generic");
        break;
    }

    ucs_error("Error on QP 0x%x wqe[%d]: %s (synd 0x%x vend 0x%x) opcode %s",
              qp_num, wqe_counter, info, ecqe->syndrome, ecqe->vendor_err_synd,
              uct_ib_mlx5_cqe_err_opcode(ecqe));
}

static unsigned uct_ib_mlx5_parse_dseg(void **dseg_p, void *qstart, void *qend,
                                       struct ibv_sge *sg, int *is_inline)
{
    struct mlx5_wqe_data_seg *dpseg;
    struct mlx5_wqe_inl_data_seg *inl;
    int ds;

    inl = *dseg_p;
    if (inl->byte_count & htonl(MLX5_INLINE_SEG)) {
        sg->addr   = (uintptr_t)(inl + 1);
        sg->lkey   = 0;
        sg->length = ntohl(inl->byte_count) & ~MLX5_INLINE_SEG;
        *is_inline = 1;
        ds         = ucs_div_round_up(sizeof(inl) + sg->length,
                                     UCT_IB_MLX5_WQE_SEG_SIZE);
    } else {
        dpseg      = *dseg_p;
        sg->addr   = ntohll(dpseg->addr);
        sg->length = ntohl(dpseg->byte_count);
        sg->lkey   = ntohl(dpseg->lkey);
        *is_inline = 0;
        ds         = 1;
    }

    *dseg_p += ds * UCT_IB_MLX5_WQE_SEG_SIZE;
    if (*dseg_p >= qend) {
        *dseg_p -= (qend - qstart);
    }
    return ds;
}

static uint64_t network_to_host(uint64_t value, int size)
{
    if (size == 4) {
        return ntohl(value);
    } else if (size == 8) {
        return ntohll(value);
    } else {
        return value;
    }
}

static void uct_ib_mlx5_wqe_dump(enum ibv_qp_type qp_type, void *wqe, void *qstart,
                                 void *qend, uct_log_data_dump_func_t packet_dump_cb,
                                 char *buffer, size_t max)
{
    static uct_ib_opcode_t opcodes[] = {
        [MLX5_OPCODE_NOP]              = { "NOP",        0 },
        [MLX5_OPCODE_RDMA_WRITE]       = { "RDMA_WRITE", UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [MLX5_OPCODE_RDMA_READ]        = { "RDMA_READ",  UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [MLX5_OPCODE_SEND]             = { "SEND",       0 },
        [MLX5_OPCODE_ATOMIC_CS]        = { "CS",         UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
        [MLX5_OPCODE_ATOMIC_FA]        = { "FA",         UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
        [MLX5_OPCODE_ATOMIC_MASKED_CS] = { "MASKED_CS",  UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC },
        [MLX5_OPCODE_ATOMIC_MASKED_FA] = { "MASKED_FA",  UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC },
   };

    struct mlx5_wqe_ctrl_seg *ctrl = wqe;
    uint8_t opcode      = ctrl->opmod_idx_opcode >> 24;
    uint8_t opmod       = ctrl->opmod_idx_opcode & 0xff;
    uint32_t qp_num     = ntohl(ctrl->qpn_ds) >> 8;
    int ds              = ctrl->qpn_ds >> 24;
    uct_ib_opcode_t *op = &opcodes[opcode];
    char *s             = buffer;
    char *ends          = buffer + max;
    struct ibv_sge sg_list[16];
    uint64_t inline_bitmap;
    int i, is_inline;
    void *seg;

    /* QP number and opcode name */
    uct_ib_log_dump_opcode(qp_num, op,
                           ctrl->fm_ce_se & MLX5_WQE_CTRL_CQ_UPDATE,
                           ctrl->fm_ce_se & MLX5_WQE_CTRL_FENCE,
                           ctrl->fm_ce_se & (1 << 1),
                           s, ends - s);
    s += strlen(s);

    /* Additional segments */
    --ds;
    seg = ctrl + 1;
    if (seg == qend) {
        seg = qstart;
    }

    /* TODO AV segment */

    /* Remote address segment */
    if (op->flags & UCT_IB_OPCODE_FLAG_HAS_RADDR) {
        struct mlx5_wqe_raddr_seg *rseg = seg;
        uct_ib_log_dump_remote_addr(ntohll(rseg->raddr), ntohl(rseg->rkey), s, ends - s);
        s += strlen(s);

        --ds;
        seg = rseg + 1;
        if (seg == qend) {
            seg = qstart;
        }
    }

    /* Atomic segment */
    if (op->flags & UCT_IB_OPCODE_FLAG_HAS_ATOMIC) {
        struct mlx5_wqe_atomic_seg *atomic = seg;
        if (opcode == MLX5_OPCODE_ATOMIC_FA) {
            uct_ib_log_dump_atomic_fadd(ntohll(atomic->swap_add), s, ends - s);
        } else if (opcode == MLX5_OPCODE_ATOMIC_CS) {
            uct_ib_log_dump_atomic_cswap(ntohll(atomic->compare),
                                     ntohll(atomic->swap_add), s, ends - s);
        }
        s += strlen(s);

        --ds;
        seg = atomic + 1;
        if (seg == qend) {
            seg = qstart;
        }
    }

    /* Extended atomic segment */
    if (op->flags & UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC) {
        uint64_t add, boundary, compare, swap, compare_mask, swap_mask;
        int size = 1 << ((opmod & 7) + 2);

        if (opcode == MLX5_OPCODE_ATOMIC_MASKED_FA) {
            add      = network_to_host(*(uint64_t*)(seg + 0), size);
            boundary = network_to_host(*(uint64_t*)(seg + size), size);
            seg     += ucs_align_up_pow2(size * 2, UCT_IB_MLX5_WQE_SEG_SIZE);
            ds      -= ucs_div_round_up(2 * size, UCT_IB_MLX5_WQE_SEG_SIZE);

            uct_ib_log_dump_atomic_masked_fadd(size, add, boundary, s, ends - s);
        } else if (opcode == MLX5_OPCODE_ATOMIC_MASKED_CS) {
            swap    = network_to_host(*(uint64_t*)(seg + 0 * size), size);
            compare = network_to_host(*(uint64_t*)(seg + 1 * size), size);

            seg += size * 2;
            if (seg == qend) {
                seg = qstart;
            }

            swap_mask    = network_to_host(*(uint64_t*)(seg + 0 * size), size);
            compare_mask = network_to_host(*(uint64_t*)(seg + 1 * size), size);
            seg += size * 2;
            if (seg == qend) {
                seg = qstart;
            }

            ucs_assert(((size * 4) % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);
            ds -= size * 4;

            uct_ib_log_dump_atomic_masked_cswap(size, compare, compare_mask, swap,
                                            swap_mask, s, ends - s);
        }
        s += strlen(s);
    }

    /* Data segments*/
    i = 0;
    inline_bitmap = 0;
    while ((ds > 0) && (i < sizeof(sg_list) / sizeof(sg_list[0]))) {
        ds -= uct_ib_mlx5_parse_dseg(&seg, qstart, qend, &sg_list[i], &is_inline);
        if (is_inline) {
            inline_bitmap |= UCS_BIT(i);
        }
        s += strlen(s);
        ++i;
    }

    uct_ib_log_dump_sg_list(sg_list, i, inline_bitmap, packet_dump_cb, s, ends - s);
}

void __uct_ib_mlx5_log_tx(const char *file, int line, const char *function,
                          enum ibv_qp_type qp_type, void *wqe, void *qstart,
                          void *qend, uct_log_data_dump_func_t packet_dump_cb)
{
    char buf[256] = {0};
    uct_ib_mlx5_wqe_dump(qp_type, wqe, qstart, qend, packet_dump_cb,
                         buf, sizeof(buf) - 1);
    uct_log_data(file, line, function, buf);
}

void __uct_ib_mlx5_log_rx(const char *file, int line, const char *function,
                          enum ibv_qp_type qp_type, struct mlx5_cqe64 *cqe, void *data,
                          uct_log_data_dump_func_t packet_dump_cb)
{
    char buf[256] = {0};
    uct_ib_log_dump_recv_completion(qp_type,
                                    ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER),
                                    ntohl(cqe->flags_rqpn) & UCS_MASK(UCT_IB_QPN_ORDER),
                                    ntohs(cqe->slid),
                                    data, ntohl(cqe->byte_cnt),
                                    packet_dump_cb, buf, sizeof(buf) - 1);
    uct_log_data(file, line, function, buf);
}

