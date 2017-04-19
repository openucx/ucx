/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_log.h"

#include <ucs/sys/sys.h>


void uct_ib_log_dump_opcode(uint32_t qp_num, uct_ib_opcode_t *op, int signal,
                        int fence, int se, char *buf, size_t max)
{
    snprintf(buf, max, "%s qp 0x%x %c%c%c", op->name, qp_num,
             signal ? 's' : '-', fence ? 'f' : '-', se ? 'e' : '-');
}

void uct_ib_log_dump_sg_list(uct_ib_iface_t *iface, uct_am_trace_type_t type,
                             struct ibv_sge *sg_list, int num_sge,
                             uint64_t inline_bitmap,
                             uct_log_data_dump_func_t data_dump,
                             char *buf, size_t max)
{
    char data[256];
    size_t total_len       = 0;
    size_t total_valid_len = 0;;
    char *s    = buf;
    char *ends = buf + max;
    void *md   = data;
    size_t len;
    int i;

    for (i = 0; i < num_sge; ++i) {
        if (inline_bitmap & UCS_BIT(i)) {
            snprintf(s, ends - s, " [inl len %d]", sg_list[i].length);
        } else {
            snprintf(s, ends - s, " [va 0x%"PRIx64" len %d lkey 0x%x]",
                     sg_list[i].addr, sg_list[i].length, sg_list[i].lkey);
        }

        len = ucs_min(sg_list[i].length, (void*)data + sizeof(data) - md);
        memcpy(md, (void*)sg_list[i].addr, len);

        s               += strlen(s);
        md              += len;
        total_len       += len;
        total_valid_len += sg_list[i].length;
    }

    if (data_dump != NULL) {
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

void uct_ib_log_dump_recv_completion(uct_ib_iface_t *iface, enum ibv_qp_type qp_type,
                                     uint32_t local_qp, uint32_t sender_qp,
                                     uint16_t sender_lid, void *data, size_t length,
                                     uct_log_data_dump_func_t data_dump,
                                     char *buf, size_t max)
{
    char *s    = buf;
    char *ends = buf + max;

    snprintf(s, ends - s, "RECV qp 0x%x", local_qp);
    s += strlen(s);

    if (qp_type == IBV_QPT_UD) {
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

static void uct_ib_dump_wr_opcode(struct ibv_qp *qp, uct_ib_opcode_t *op,
                                  int send_flags, char *buf, size_t max)
{
    uct_ib_log_dump_opcode(qp->qp_num, op,
                           send_flags & IBV_SEND_SIGNALED,
                           send_flags & IBV_SEND_FENCE,
                           send_flags & IBV_SEND_SOLICITED,
                           buf, max);
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
        s += strlen(s);
    }
}

static void uct_ib_dump_send_wr(uct_ib_iface_t *iface, struct ibv_qp *qp,
                                struct ibv_send_wr *wr,
                                uct_log_data_dump_func_t data_dump,
                                char *buf, size_t max)
{
    static uct_ib_opcode_t opcodes[] = {
        [IBV_WR_RDMA_WRITE]           = { "RDMA_WRITE", UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [IBV_WR_RDMA_READ]            = { "RDMA_READ",  UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [IBV_WR_SEND]                 = { "SEND",       0 },
        [IBV_WR_ATOMIC_CMP_AND_SWP]   = { "CS",         UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
        [IBV_WR_ATOMIC_FETCH_AND_ADD] = { "FA",         UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
   };

    char *s             = buf;
    char *ends          = buf + max;
    uct_ib_opcode_t *op = &opcodes[wr->opcode];

    uct_ib_dump_wr_opcode(qp, op, wr->send_flags, s, ends - s);
    s += strlen(s);

    uct_ib_dump_wr(qp, op, wr, s, ends - s);
    s += strlen(s);

    uct_ib_log_dump_sg_list(iface, UCT_AM_TRACE_TYPE_SEND, wr->sg_list, wr->num_sge,
                            (wr->send_flags & IBV_SEND_INLINE) ? -1 : 0,
                            data_dump, s, ends - s);
}

void __uct_ib_log_post_send(const char *file, int line, const char *function,
                            uct_ib_iface_t *iface, struct ibv_qp *qp,
                            struct ibv_send_wr *wr,
                            uct_log_data_dump_func_t data_dump_cb)
{
    char buf[256] = {0};
    while (wr != NULL) {
        uct_ib_dump_send_wr(iface, qp, wr, data_dump_cb, buf, sizeof(buf) - 1);
        uct_log_data(file, line, function, buf);
        wr = wr->next;
    }
}

void __uct_ib_log_recv_completion(const char *file, int line, const char *function,
                                  uct_ib_iface_t *iface, enum ibv_qp_type qp_type,
                                  uint32_t l_qp, uint32_t r_qp, uint16_t slid,
                                  void *data, size_t length,
                                  uct_log_data_dump_func_t packet_dump_cb)
{
    char buf[256] = {0};
    size_t len;

    len = length;
    if (qp_type == IBV_QPT_UD) {
        len  -= UCT_IB_GRH_LEN;
        data += UCT_IB_GRH_LEN;
    }
    uct_ib_log_dump_recv_completion(iface, qp_type, l_qp, r_qp, slid, data, len,
                                    packet_dump_cb, buf, sizeof(buf) - 1);
    uct_log_data(file, line, function, buf);
}

#if HAVE_DECL_IBV_EXP_POST_SEND
static void uct_ib_dump_exp_send_wr(uct_ib_iface_t *iface, struct ibv_qp *qp,
                                    struct ibv_exp_send_wr *wr,
                                    uct_log_data_dump_func_t data_dump_cb,
                                    char *buf, size_t max)
{
    static uct_ib_opcode_t exp_opcodes[] = {
#if HAVE_DECL_IBV_EXP_WR_NOP
        [IBV_EXP_WR_NOP]                  = { "NOP",        0},
#endif
        [IBV_EXP_WR_RDMA_WRITE]           = { "RDMA_WRITE", UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [IBV_EXP_WR_RDMA_READ]            = { "RDMA_READ",  UCT_IB_OPCODE_FLAG_HAS_RADDR },
        [IBV_EXP_WR_SEND]                 = { "SEND",       0 },
        [IBV_EXP_WR_ATOMIC_CMP_AND_SWP]   = { "CS",         UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
        [IBV_EXP_WR_ATOMIC_FETCH_AND_ADD] = { "FA",         UCT_IB_OPCODE_FLAG_HAS_ATOMIC },
#if HAVE_DECL_IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP
        [IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP]   = { "MASKED_CS", UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC },
#endif
#if HAVE_DECL_IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD
        [IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD] = { "MASKED_FA", UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC },
#endif
   };

   char *s    = buf;
   char *ends = buf + max;
   uct_ib_opcode_t *op = &exp_opcodes[wr->exp_opcode];

   /* opcode in legacy mode */
   UCS_STATIC_ASSERT((int)IBV_SEND_SIGNALED  == (int)IBV_EXP_SEND_SIGNALED);
   UCS_STATIC_ASSERT((int)IBV_SEND_FENCE     == (int)IBV_EXP_SEND_FENCE);
   UCS_STATIC_ASSERT((int)IBV_SEND_SOLICITED == (int)IBV_EXP_SEND_SOLICITED);
   uct_ib_dump_wr_opcode(qp, op, wr->exp_send_flags, s, ends - s);
   s += strlen(s);

   /* TODO DC address handle */

   /* WR data in legacy mode */
   UCS_STATIC_ASSERT((int)IBV_WR_ATOMIC_FETCH_AND_ADD ==  (int)IBV_EXP_WR_ATOMIC_FETCH_AND_ADD);
   UCS_STATIC_ASSERT((int)IBV_WR_ATOMIC_CMP_AND_SWP   ==  (int)IBV_EXP_WR_ATOMIC_CMP_AND_SWP);
   UCS_STATIC_ASSERT(ucs_offsetof(struct ibv_send_wr, opcode) ==
                     ucs_offsetof(struct ibv_exp_send_wr, exp_opcode));
   UCS_STATIC_ASSERT(ucs_offsetof(struct ibv_send_wr, wr) ==
                     ucs_offsetof(struct ibv_exp_send_wr, wr));
   uct_ib_dump_wr(qp, op, (struct ibv_send_wr*)wr, s, ends - s);
   s += strlen(s);

   /* Extended atomics */
#if HAVE_IB_EXT_ATOMICS
   if (op->flags & UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC) {
       uct_ib_log_dump_remote_addr(wr->ext_op.masked_atomics.remote_addr,
                             wr->ext_op.masked_atomics.rkey,
                             s, ends - s);
       s += strlen(s);

       if (wr->exp_opcode == IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD) {
           uct_ib_log_dump_atomic_masked_fadd(wr->ext_op.masked_atomics.log_arg_sz,
                                          wr->ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.add_val,
                                          wr->ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.field_boundary,
                                          s, ends - s);
       } else if (wr->exp_opcode == IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP) {
           uct_ib_log_dump_atomic_masked_cswap(wr->ext_op.masked_atomics.log_arg_sz,
                                           wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_val,
                                           wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_mask,
                                           wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_val,
                                           wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_mask,
                                           s, ends - s);
       }
       s += strlen(s);
   }
#endif

   uct_ib_log_dump_sg_list(iface, UCT_AM_TRACE_TYPE_SEND, wr->sg_list, wr->num_sge,
                           (wr->exp_send_flags & IBV_EXP_SEND_INLINE) ? -1 : 0,
                           data_dump_cb, s, ends - s);
}

void __uct_ib_log_exp_post_send(const char *file, int line, const char *function,
                                uct_ib_iface_t *iface, struct ibv_qp *qp,
                                struct ibv_exp_send_wr *wr,
                                uct_log_data_dump_func_t packet_dump_cb)
{
    char buf[256] = {0};
    while (wr != NULL) {
        uct_ib_dump_exp_send_wr(iface, qp, wr, packet_dump_cb, buf, sizeof(buf) - 1);
        uct_log_data(file, line, function, buf);
        wr = wr->next;
    }
}

#endif
