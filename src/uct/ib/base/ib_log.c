/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ib_log.h"

#include <ucs/sys/sys.h>


const char *uct_ib_qp_type_str(int qp_type)
{
    switch (qp_type) {
    case IBV_QPT_RC:
        return "RC";
    case IBV_QPT_UD:
        return "UD";
#if HAVE_TL_DC
    case UCT_IB_QPT_DCI:
        return "DCI";
#endif
    default:
        ucs_bug("invalid qp type: %d", qp_type);
        return "unknown";
    }
}

void uct_ib_log_dump_opcode(uct_ib_opcode_t *op, int sig, int fence, int se,
                            char *buf, size_t max)
{
    snprintf(buf, max, "%s %c%c%c", op->name,
             sig    ? 's' : '-',
             fence  ? 'f' : '-',
             se     ? 'e' : '-');
}

void uct_ib_log_dump_sg_list(uct_ib_iface_t *iface, uct_am_trace_type_t type,
                             const char *sg_prefixes, struct ibv_sge *sg_list,
                             int num_sge, uint64_t inline_bitmap,
                             uct_log_data_dump_func_t data_dump,
                             int data_dump_sge, char *buf, size_t max)
{
    char data[256];
    size_t total_len       = 0;
    size_t total_valid_len = 0;
    char *s    = buf;
    char *ends = buf + max;
    void *md   = data;
    size_t len;
    int i;

    for (i = 0; i < num_sge; ++i) {
        if (inline_bitmap & UCS_BIT(i)) {
            snprintf(s, ends - s, " [inl len %d]", sg_list[i].length);
        } else {
            snprintf(s, ends - s, " %.1s[va 0x%"PRIx64" len %d lkey 0x%x]",
                     (sg_prefixes == NULL) ? "" : &sg_prefixes[i],
                     sg_list[i].addr, sg_list[i].length, sg_list[i].lkey);
        }

        s               += strlen(s);

        if ((i < data_dump_sge) && data_dump) {
            len = ucs_min(sg_list[i].length,
                          UCS_PTR_BYTE_DIFF(md, data) + sizeof(data));
            memcpy(md, (void*)sg_list[i].addr, len);

            md               = UCS_PTR_BYTE_OFFSET(md, len);
            total_len       += len;
            total_valid_len += sg_list[i].length;
        }
    }

    if (data_dump) {
        data_dump(&iface->super, type, data, total_len, total_valid_len, s, ends - s);
    }
}

void uct_ib_log_dump_remote_addr(uint64_t remote_addr, uint32_t rkey,
                                 char *buf, size_t max)
{
    snprintf(buf, max, " [rva 0x%"PRIx64" rkey 0x%x]", remote_addr, rkey);
}

void uct_ib_log_dump_atomic_fadd(uint64_t add, char *buf, size_t max)
{
    snprintf(buf, max, " [add %ld]", add);
}

void uct_ib_log_dump_atomic_cswap(uint64_t compare, uint64_t swap, char *buf, size_t max)
{
    snprintf(buf, max, " [cmp %ld swap %ld]", compare, swap);
}

void uct_ib_log_dump_atomic_masked_fadd(int argsize, uint64_t add, uint64_t boundary,
                                        char *buf, size_t max)
{
    snprintf(buf, max, " [%dbit add %"PRIi64"/0x%"PRIx64"]", argsize * 8, add, boundary);
}

void uct_ib_log_dump_atomic_masked_cswap(int argsize, uint64_t compare, uint64_t compare_mask,
                                         uint64_t swap, uint64_t swap_mask,
                                         char *buf, size_t max)
{
    snprintf(buf, max, " [%d bit cmp %"PRIi64"/0x%"PRIx64" swap %"PRIi64"/0x%"PRIx64"]",
             argsize * 8, compare, compare_mask, swap, swap_mask);
}

void uct_ib_log_dump_qp_peer_info(uct_ib_iface_t *iface,
                                  const struct ibv_ah_attr *ah_attr,
                                  uint32_t dest_qpn, char *buf, size_t max)
{
    char *s    = buf;
    char *ends = buf + max;

    snprintf(s, ends - s, "[rqpn 0x%x ", dest_qpn);
    s += strlen(s);

    uct_ib_ah_attr_str(s, ends - s, ah_attr);
    s += strlen(s);

    snprintf(s, ends - s, "]");
}

void uct_ib_log_dump_recv_completion(uct_ib_iface_t *iface, uint32_t local_qp,
                                     uint32_t sender_qp, uint16_t sender_lid,
                                     void *data, size_t length,
                                     uct_log_data_dump_func_t data_dump,
                                     char *buf, size_t max)
{
    char *s    = buf;
    char *ends = buf + max;

    snprintf(s, ends - s, "RECV qp 0x%x", local_qp);
    s += strlen(s);

    if (iface->config.qp_type == IBV_QPT_UD) {
        snprintf(s, ends - s, " [slid %d sqp 0x%x]", sender_lid, sender_qp);
        s += strlen(s);
    }

    snprintf(s, ends - s, " [va %p len %zu]", data, length);
    s += strlen(s);

    if (data_dump != NULL) {
        data_dump(&iface->super, UCT_AM_TRACE_TYPE_RECV, data, length, length,
                  s, ends - s);
    }
}

static void uct_ib_dump_wr_opcode(struct ibv_qp *qp, uint64_t wr_id,
                                  uct_ib_opcode_t *op, int send_flags,
                                  char *buf, size_t max)
{
    char *s    = buf;
    char *ends = buf + max;

    snprintf(s, ends - s, "QP 0x%x wrid 0x%"PRIx64" ", qp->qp_num, wr_id);
    s += strlen(s);

    uct_ib_log_dump_opcode(op,
                           send_flags & IBV_SEND_SIGNALED,
                           send_flags & IBV_SEND_FENCE,
                           send_flags & IBV_SEND_SOLICITED,
                           s, ends - s);
}

static void uct_ib_dump_wr(struct ibv_qp *qp, uct_ib_opcode_t *op,
                           struct ibv_send_wr *wr, char *buf, size_t max)
{
    char *s    = buf;
    char *ends = buf + max;

    if (qp->qp_type == IBV_QPT_UD) {
        snprintf(s, ends - s, " [rqpn 0x%x ah %p]", wr->wr.ud.remote_qpn,
                 wr->wr.ud.ah);
        s += strlen(s);
    }

    if (op->flags & UCT_IB_OPCODE_FLAG_HAS_RADDR) {
        uct_ib_log_dump_remote_addr(wr->wr.rdma.remote_addr, wr->wr.rdma.rkey,
                                    s, ends - s);
        s += strlen(s);
    }

    if (op->flags & UCT_IB_OPCODE_FLAG_HAS_ATOMIC) {
        uct_ib_log_dump_remote_addr(wr->wr.atomic.remote_addr, wr->wr.atomic.rkey,
                                    s, ends - s);
        s += strlen(s);

        if (wr->opcode == IBV_WR_ATOMIC_FETCH_AND_ADD) {
            uct_ib_log_dump_atomic_fadd(wr->wr.atomic.compare_add, s, ends - s);
        } else if (wr->opcode == IBV_WR_ATOMIC_CMP_AND_SWP) {
            uct_ib_log_dump_atomic_cswap(wr->wr.atomic.compare_add,
                                     wr->wr.atomic.swap, s, ends - s);
        }

        /* do not forget `s += strlen(s);` here if you are
         * processing more information for dumping below */
    }
}

static void uct_ib_dump_send_wr(uct_ib_iface_t *iface, struct ibv_qp *qp,
                                struct ibv_send_wr *wr, int max_sge,
                                uct_log_data_dump_func_t data_dump,
                                char *buf, size_t max)
{
    static uct_ib_opcode_t opcodes[] = {
        [IBV_WR_RDMA_WRITE]           = { "RDMA_WRITE", UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [IBV_WR_RDMA_READ]            = { "RDMA_READ",  UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [IBV_WR_SEND]                 = { "SEND",       0 },
        [IBV_WR_SEND_WITH_IMM]        = { "SEND_IMM",   0 },
        [IBV_WR_ATOMIC_CMP_AND_SWP]   = { "CSWAP",      UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
        [IBV_WR_ATOMIC_FETCH_AND_ADD] = { "FETCH_ADD",  UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
    };

    char *s             = buf;
    char *ends          = buf + max;
    uct_ib_opcode_t *op = &opcodes[wr->opcode];

    uct_ib_dump_wr_opcode(qp, wr->wr_id, op, wr->send_flags, s, ends - s);
    s += strlen(s);

    uct_ib_dump_wr(qp, op, wr, s, ends - s);
    s += strlen(s);

    uct_ib_log_dump_sg_list(iface, UCT_AM_TRACE_TYPE_SEND, NULL, wr->sg_list,
                            wr->num_sge,
                            (wr->send_flags & IBV_SEND_INLINE) ? -1 : 0,
                            data_dump, max_sge, s, ends - s);
}

void uct_ib_memlock_limit_msg(ucs_string_buffer_t *message, int sys_errno)
{
    size_t memlock_limit;
    ucs_status_t status;

    if (sys_errno == ENOMEM) {
        status = ucs_sys_get_effective_memlock_rlimit(&memlock_limit);
        if ((status == UCS_OK) && (memlock_limit != SIZE_MAX)) {
            ucs_string_buffer_appendf(
                    message,
                    " : Please set max locked memory (ulimit -l) to 'unlimited'"
                    " (current: %llu kbytes)",
                    memlock_limit / UCS_KBYTE);
        }
    }
}

void __uct_ib_log_post_send(const char *file, int line, const char *function,
                            uct_ib_iface_t *iface, struct ibv_qp *qp,
                            struct ibv_send_wr *wr, int max_sge,
                            uct_log_data_dump_func_t data_dump_cb)
{
    char buf[256] = {0};
    while (wr != NULL) {
        uct_ib_dump_send_wr(iface, qp, wr, max_sge, data_dump_cb, buf, sizeof(buf) - 1);
        uct_log_data(file, line, function, buf);
        wr = wr->next;
    }
}

void __uct_ib_log_recv_completion(const char *file, int line, const char *function,
                                  uct_ib_iface_t *iface, uint32_t l_qp,
                                  uint32_t r_qp, uint16_t slid, void *data,
                                  size_t length,
                                  uct_log_data_dump_func_t packet_dump_cb)
{
    char buf[256] = {0};
    size_t len;

    len = length;
    if (iface->config.qp_type == IBV_QPT_UD) {
        len  -= UCT_IB_GRH_LEN;
        data  = UCS_PTR_BYTE_OFFSET(data, UCT_IB_GRH_LEN);
    }
    uct_ib_log_dump_recv_completion(iface, l_qp, r_qp, slid, data, len,
                                    packet_dump_cb, buf, sizeof(buf) - 1);
    uct_log_data(file, line, function, buf);
}
