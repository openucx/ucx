/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ib_mlx5_log.h"

#include <uct/ib/base/ib_device.h>
#include <string.h>


static void uct_ib_mlx5_wqe_dump(uct_ib_iface_t *iface, void *wqe, void *qstart,
                                 void *qend, int max_sge, int dump_qp,
                                 uct_log_data_dump_func_t packet_dump_cb,
                                 char *buffer, size_t max, uct_ib_log_sge_t *log_sge);

static const char *uct_ib_mlx5_cqe_err_opcode(uct_ib_mlx5_err_cqe_t *ecqe)
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
            return "CSWAP";
        case MLX5_OPCODE_ATOMIC_FA:
            return "FETCH_ADD";
        case MLX5_OPCODE_ATOMIC_MASKED_CS:
            return "MASKED_CSWAP";
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

ucs_status_t uct_ib_mlx5_completion_with_err(uct_ib_iface_t *iface,
                                             uct_ib_mlx5_err_cqe_t *ecqe,
                                             uct_ib_mlx5_txwq_t *txwq,
                                             ucs_log_level_t log_level)
{
    ucs_status_t status        = UCS_ERR_IO_ERROR;
    char         err_info[256] = {};
    char         wqe_info[256] = {};
    uint16_t     wqe_index;
    uint32_t     qp_num;
    void         *wqe;

    wqe_index = ntohs(ecqe->wqe_counter);
    qp_num    = ntohl(ecqe->s_wqe_opcode_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    if (txwq != NULL) {
        wqe_index %= UCS_PTR_BYTE_DIFF(txwq->qstart, txwq->qend) / MLX5_SEND_WQE_BB;
    }

    if (ecqe->syndrome == MLX5_CQE_SYNDROME_WR_FLUSH_ERR) {
        ucs_trace("QP 0x%x wqe[%d] is flushed", qp_num, wqe_index);
        return status;
    }

    switch (ecqe->syndrome) {
    case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR:
        snprintf(err_info, sizeof(err_info), "Local length");
        break;
    case MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR:
        snprintf(err_info, sizeof(err_info), "Local QP operation");
        break;
    case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR:
        snprintf(err_info, sizeof(err_info), "Local protection");
        break;
    case MLX5_CQE_SYNDROME_WR_FLUSH_ERR:
        snprintf(err_info, sizeof(err_info), "WR flushed because QP in error state");
        break;
    case MLX5_CQE_SYNDROME_MW_BIND_ERR:
        snprintf(err_info, sizeof(err_info), "Memory window bind");
        break;
    case MLX5_CQE_SYNDROME_BAD_RESP_ERR:
        snprintf(err_info, sizeof(err_info), "Bad response");
        break;
    case MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR:
        snprintf(err_info, sizeof(err_info), "Local access");
        break;
    case MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR:
        snprintf(err_info, sizeof(err_info), "Invalid request");
        break;
    case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR:
        snprintf(err_info, sizeof(err_info), "Remote access");
        status = UCS_ERR_CONNECTION_RESET;
        break;
    case MLX5_CQE_SYNDROME_REMOTE_OP_ERR:
        snprintf(err_info, sizeof(err_info), "Remote QP");
        status = UCS_ERR_CONNECTION_RESET;
        break;
    case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR:
        snprintf(err_info, sizeof(err_info), "Transport retry count exceeded");
        status = UCS_ERR_ENDPOINT_TIMEOUT;
        break;
    case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR:
        snprintf(err_info, sizeof(err_info), "Receive-no-ready retry count exceeded");
        status = UCS_ERR_ENDPOINT_TIMEOUT;
        break;
    case MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR:
        snprintf(err_info, sizeof(err_info), "Remote side aborted");
        status = UCS_ERR_ENDPOINT_TIMEOUT;
        break;
    default:
        snprintf(err_info, sizeof(err_info), "Generic");
        break;
    }

    if ((txwq != NULL) && ((ecqe->op_own >> 4) == MLX5_CQE_REQ_ERR)) {
        wqe = UCS_PTR_BYTE_OFFSET(txwq->qstart, MLX5_SEND_WQE_BB * wqe_index);
        uct_ib_mlx5_wqe_dump(iface, wqe, txwq->qstart, txwq->qend, INT_MAX, 0,
                             NULL, wqe_info, sizeof(wqe_info) - 1, NULL);
    } else {
        snprintf(wqe_info, sizeof(wqe_info) - 1, "opcode %s",
                 uct_ib_mlx5_cqe_err_opcode(ecqe));
    }

    ucs_log(log_level,
            "%s on "UCT_IB_IFACE_FMT"/%s (synd 0x%x vend 0x%x hw_synd %d/%d)\n"
            "%s QP 0x%x wqe[%d]: %s",
            err_info, UCT_IB_IFACE_ARG(iface),
            uct_ib_iface_is_roce(iface) ? "RoCE" : "IB",
            ecqe->syndrome, ecqe->vendor_err_synd, ecqe->hw_synd_type >> 4,
            ecqe->hw_err_synd, uct_ib_qp_type_str(iface->config.qp_type),
            qp_num, wqe_index, wqe_info);
    return status;
}

static unsigned uct_ib_mlx5_parse_dseg(void **dseg_p, void *qstart, void *qend,
                                       struct ibv_sge *sg_list, int *sg_index,
                                       int *is_inline)
{
    struct mlx5_wqe_data_seg *dpseg;
    struct mlx5_wqe_inl_data_seg *inl;
    struct ibv_sge *sg = &sg_list[*sg_index];
    int byte_count;
    void *addr;
    int ds;

    if (*dseg_p == qend) {
        *dseg_p = qstart;
    }
    inl = *dseg_p;
    if (inl->byte_count & htonl(MLX5_INLINE_SEG)) {
        addr       = inl + 1;
        sg->addr   = (uintptr_t)addr;
        sg->lkey   = 0;
        byte_count = ntohl(inl->byte_count) & ~MLX5_INLINE_SEG;
        if (UCS_PTR_BYTE_OFFSET(addr, byte_count) > qend) {
            sg->length       = UCS_PTR_BYTE_DIFF(addr, qend);
            (sg + 1)->addr   = (uintptr_t)qstart;
            (sg + 1)->lkey   = 0;
            (sg + 1)->length = byte_count - sg->length;
            ++(*sg_index);
        } else {
            sg->length       = byte_count;
        }
        *is_inline = 1;
        ds         = ucs_div_round_up(sizeof(*inl) + byte_count,
                                     UCT_IB_MLX5_WQE_SEG_SIZE);
        ++(*sg_index);
    } else {
        dpseg      = *dseg_p;
        sg->addr   = be64toh(dpseg->addr);
        sg->length = ntohl(dpseg->byte_count);
        sg->lkey   = ntohl(dpseg->lkey);
        *is_inline = 0;
        ds         = 1;
        ++(*sg_index);
    }

    *dseg_p = UCS_PTR_BYTE_OFFSET(*dseg_p, ds * UCT_IB_MLX5_WQE_SEG_SIZE);
    if (*dseg_p >= qend) {
        *dseg_p = UCS_PTR_BYTE_OFFSET(*dseg_p, -UCS_PTR_BYTE_DIFF(qstart, qend));
    }
    return ds;
}

static uint64_t network_to_host(void *ptr, int size)
{
    if (size == 4) {
        return ntohl(*(uint32_t*)ptr);
    } else if (size == 8) {
        return be64toh(*(uint64_t*)ptr);
    } else {
        return *(uint64_t*)ptr;
    }
}

static size_t uct_ib_mlx5_dump_dgram(char *buf, size_t max, void *seg, int is_eth)
{
    struct mlx5_wqe_datagram_seg *dgseg = seg;
    struct mlx5_base_av *av;
    struct mlx5_grh_av *grh_av;
    uct_ib_mlx5_base_av_t base_av;

    av     = mlx5_av_base(&dgseg->av);
    grh_av = mlx5_av_grh(&dgseg->av);

    UCT_IB_MLX5_SET_BASE_AV(&base_av, av);
    uct_ib_mlx5_av_dump(buf, max, &base_av, grh_av, is_eth);

    return (base_av.dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV) ?
           UCT_IB_MLX5_AV_FULL_SIZE : UCT_IB_MLX5_AV_BASE_SIZE;
}

static int uct_ib_mlx5_is_qp_require_av_seg(int qp_type)
{
    if (qp_type == IBV_QPT_UD) {
        return 1;
    }
#if HAVE_TL_DC
    if (qp_type == UCT_IB_QPT_DCI) {
        return 1;
    }
#endif
    return 0;
}

static void uct_ib_mlx5_wqe_dump(uct_ib_iface_t *iface, void *wqe, void *qstart,
                                 void *qend, int max_sge, int dump_qp,
                                 uct_log_data_dump_func_t packet_dump_cb,
                                 char *buffer, size_t max, uct_ib_log_sge_t *log_sge)
{
    static uct_ib_opcode_t opcodes[] = {
        [MLX5_OPCODE_NOP]              = { "NOP",        0 },
        [MLX5_OPCODE_RDMA_WRITE]       = { "RDMA_WRITE", UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [MLX5_OPCODE_RDMA_READ]        = { "RDMA_READ",  UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [MLX5_OPCODE_SEND]             = { "SEND",       0 },
        [MLX5_OPCODE_SEND_IMM]         = { "SEND_IMM",   0 },
        [MLX5_OPCODE_ATOMIC_CS]        = { "CSWAP",      UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
        [MLX5_OPCODE_ATOMIC_FA]        = { "FETCH_ADD",  UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
        [MLX5_OPCODE_ATOMIC_MASKED_CS] = { "MASKED_CSWAP",
                                           UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC },
        [MLX5_OPCODE_ATOMIC_MASKED_FA] = { "MASKED_FETCH_ADD",
                                           UCT_IB_OPCODE_FLAG_HAS_RADDR|UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC },
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
    int i, is_inline, is_eth;
    size_t dg_size;
    void *seg;

    /* QP and WQE index */
    if (dump_qp) {
        snprintf(s, ends - s, "QP 0x%x [%03ld] ", qp_num,
                 UCS_PTR_BYTE_DIFF(qstart, wqe) / MLX5_SEND_WQE_BB);
        s += strlen(s);
    }

    /* Opcode and flags */
    uct_ib_log_dump_opcode(op,
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

    if (uct_ib_mlx5_is_qp_require_av_seg(iface->config.qp_type)) {
        is_eth = uct_ib_iface_is_roce(iface);
        dg_size = uct_ib_mlx5_dump_dgram(s, ends - s, seg, is_eth);
        s += strlen(s);

        seg = (char *)seg + dg_size;
        ds -= ucs_div_round_up(dg_size, UCT_IB_MLX5_WQE_SEG_SIZE);
    }
    if (seg == qend) {
        seg = qstart;
    }

    /* Remote address segment */
    if (op->flags & UCT_IB_OPCODE_FLAG_HAS_RADDR) {
        struct mlx5_wqe_raddr_seg *rseg = seg;
        uct_ib_log_dump_remote_addr(be64toh(rseg->raddr), ntohl(rseg->rkey), s, ends - s);
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
            uct_ib_log_dump_atomic_fadd(be64toh(atomic->swap_add), s, ends - s);
        } else if (opcode == MLX5_OPCODE_ATOMIC_CS) {
            uct_ib_log_dump_atomic_cswap(be64toh(atomic->compare),
                                         be64toh(atomic->swap_add), s, ends - s);
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
            add      = network_to_host(seg, size);
            boundary = network_to_host(UCS_PTR_BYTE_OFFSET(seg, size), size);
            seg      = UCS_PTR_BYTE_OFFSET(seg,
                                           ucs_align_up_pow2(size * 2,
                                                             UCT_IB_MLX5_WQE_SEG_SIZE));
            ds      -= ucs_div_round_up(2 * size, UCT_IB_MLX5_WQE_SEG_SIZE);

            uct_ib_log_dump_atomic_masked_fadd(size, add, boundary, s, ends - s);
        } else if (opcode == MLX5_OPCODE_ATOMIC_MASKED_CS) {
            swap    = network_to_host(seg, size);
            compare = network_to_host(UCS_PTR_BYTE_OFFSET(seg, size), size);

            seg = UCS_PTR_BYTE_OFFSET(seg, size * 2);
            if (seg == qend) {
                seg = qstart;
            }

            swap_mask    = network_to_host(seg, size);
            compare_mask = network_to_host(UCS_PTR_BYTE_OFFSET(seg, size), size);
            seg          = UCS_PTR_BYTE_OFFSET(seg, size * 2);
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
    if (log_sge == NULL) {
        i = 0;
        inline_bitmap = 0;

        while ((ds > 0) && (i < sizeof(sg_list) / sizeof(sg_list[0]))) {
            ds -= uct_ib_mlx5_parse_dseg(&seg, qstart, qend, sg_list, &i, &is_inline);
            if (is_inline) {
                inline_bitmap |= UCS_BIT(i-1);
            }
            s += strlen(s);
        }
    }

    uct_ib_log_dump_sg_list(iface, UCT_AM_TRACE_TYPE_SEND,
                            log_sge ? log_sge->sg_list : sg_list,
                            log_sge ? log_sge->num_sge : ucs_min(i, max_sge),
                            log_sge ? log_sge->inline_bitmap : inline_bitmap,
                            packet_dump_cb, s, ends - s);
}

void __uct_ib_mlx5_log_tx(const char *file, int line, const char *function,
                          uct_ib_iface_t *iface, void *wqe, void *qstart,
                          void *qend, int max_sge, uct_ib_log_sge_t *log_sge,
                          uct_log_data_dump_func_t packet_dump_cb)
{
    char buf[256] = {0};
    uct_ib_mlx5_wqe_dump(iface, wqe, qstart, qend, max_sge, 1, packet_dump_cb,
                         buf, sizeof(buf) - 1, log_sge);
    uct_log_data(file, line, function, buf);
}

void uct_ib_mlx5_cqe_dump(const char *file, int line, const char *function, struct mlx5_cqe64 *cqe)
{
    char buf[256] = {0};

    snprintf(buf, sizeof(buf) - 1,
            "CQE(op_own 0x%x) qp 0x%x sqp 0x%x slid %d bytes %d wqe_idx %d ",
            (unsigned)cqe->op_own,
            (unsigned)(ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER)),
            (unsigned)(ntohl(cqe->flags_rqpn) & UCS_MASK(UCT_IB_QPN_ORDER)),
            (unsigned)ntohs(cqe->slid),
            (unsigned)ntohl(cqe->byte_cnt),
            (unsigned)ntohs(cqe->wqe_counter));

    uct_log_data(file, line, function, buf);
}

void uct_ib_mlx5_av_dump(char *buf, size_t max,
                         const uct_ib_mlx5_base_av_t *base_av,
                         const struct mlx5_grh_av *grh_av, int is_eth)
{
    char gid_buf[32];
    int sgid_index;
    char *p, *endp;

    p    = buf;
    endp = buf + max - 1;

    snprintf(p, endp - p, " [rqpn 0x%x",
             ntohl(base_av->dqp_dct & ~UCT_IB_MLX5_EXTENDED_UD_AV));
    p += strlen(p);

    if (!is_eth) {
        snprintf(p, endp - p, " rlid %d", ntohs(base_av->rlid));
        p += strlen(p);
    }

    if (base_av->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV) {
        if (is_eth || (grh_av->grh_gid_fl & UCT_IB_MLX5_AV_GRH_PRESENT)) {
            if (is_eth) {
                snprintf(p, endp - p, " rmac %02x:%02x:%02x:%02x:%02x:%02x",
                         grh_av->rmac[0], grh_av->rmac[1], grh_av->rmac[2],
                         grh_av->rmac[3], grh_av->rmac[4], grh_av->rmac[5]);
                p += strlen(p);
            }

            sgid_index = (htonl(grh_av->grh_gid_fl) >> 20) & UCS_MASK(8);
            snprintf(p, endp - p,  " sgix %d dgid %s tc %d]", sgid_index,
                     uct_ib_gid_str((union ibv_gid *)grh_av->rgid, gid_buf,
                                    sizeof(gid_buf)),
                     grh_av->tclass);
        } else {
            snprintf(p, endp - p, "]");
        }
    } else {
        snprintf(p, endp - p, "]");
    }
}

void __uct_ib_mlx5_log_rx(const char *file, int line, const char *function,
                          uct_ib_iface_t *iface, struct mlx5_cqe64 *cqe,
                          void *data, uct_log_data_dump_func_t packet_dump_cb)
{
    char buf[256] = {0};
    size_t length;

    length = ntohl(cqe->byte_cnt) & UCT_IB_MLX5_MP_RQ_BYTE_CNT_MASK;
    if (iface->config.qp_type == IBV_QPT_UD) {
        length -= UCT_IB_GRH_LEN;
        data    = UCS_PTR_BYTE_OFFSET(data, UCT_IB_GRH_LEN);
    }
    uct_ib_log_dump_recv_completion(iface,
                                    ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER),
                                    ntohl(cqe->flags_rqpn) & UCS_MASK(UCT_IB_QPN_ORDER),
                                    ntohs(cqe->slid),
                                    data, length,
                                    packet_dump_cb, buf, sizeof(buf) - 1);
    uct_log_data(file, line, function, buf);
}

