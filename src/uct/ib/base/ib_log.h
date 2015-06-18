/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_LOG_H
#define UCT_IB_LOG_H

#include "ib_verbs.h"

#include <uct/tl/tl_log.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>


enum {
    UCT_IB_OPCODE_FLAG_HAS_RADDR       = UCS_BIT(0),
    UCT_IB_OPCODE_FLAG_HAS_ATOMIC      = UCS_BIT(1),
    UCT_IB_OPCODE_FLAG_HAS_EXT_ATOMIC  = UCS_BIT(2)
};


typedef struct uct_ib_opcode {
    const char *name;
    uint32_t   flags;
} uct_ib_opcode_t;


void uct_ib_log_dump_opcode(uint32_t qp_num, uct_ib_opcode_t *op, int signal,
                            int fence, int se, char *buf, size_t max);

void uct_ib_log_dump_sg_list(struct ibv_sge *sg_list, int num_sge, uint64_t inline_bitmap,
                             uct_log_data_dump_func_t data_dump, char *buf, size_t max);

void uct_ib_log_dump_remote_addr(uint64_t remote_addr, uint32_t rkey,
                                 char *buf, size_t max);

void uct_ib_log_dump_atomic_fadd(uint64_t add, char *buf, size_t max);

void uct_ib_log_dump_atomic_cswap(uint64_t compare, uint64_t swap, char *buf, size_t max);

void uct_ib_log_dump_atomic_masked_fadd(int argsize, uint64_t add, uint64_t boundary,
                                        char *buf, size_t max);

void uct_ib_log_dump_atomic_masked_cswap(int argsize, uint64_t compare, uint64_t compare_mask,
                                         uint64_t swap, uint64_t swap_mask,
                                         char *buf, size_t max);

void uct_ib_log_dump_recv_completion(enum ibv_qp_type qp_type,
                                     uint32_t local_qp, uint32_t sender_qp,
                                     uint16_t sender_lid, void *data, size_t length,
                                     uct_log_data_dump_func_t data_dump,
                                     char *buf, size_t max);

void __uct_ib_log_post_send(const char *file, int line, const char *function,
                            struct ibv_qp *qp, struct ibv_send_wr *wr,
                            uct_log_data_dump_func_t packet_dump_cb);

void __uct_ib_log_recv_completion(const char *file, int line, const char *function,
                                  enum ibv_qp_type qp_type, struct ibv_wc *wc, void *data,
                                  uct_log_data_dump_func_t packet_dump_cb);

#if HAVE_DECL_IBV_EXP_POST_SEND
void __uct_ib_log_exp_post_send(const char *file, int line, const char *function,
                                struct ibv_qp *qp, struct ibv_exp_send_wr *wr,
                                uct_log_data_dump_func_t packet_dump_cb);
#endif


#define uct_ib_log_post_send(_qp, _wr, _dump_cb) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_log_post_send(__FILE__, __LINE__, __FUNCTION__, _qp, _wr, _dump_cb); \
    }

#define uct_ib_log_recv_completion(_qp_type, _wc, _data, _dump_cb, ...) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_log_recv_completion(__FILE__, __LINE__, __FUNCTION__, \
                                     _qp_type, _wc, _data, _dump_cb, ## __VA_ARGS__); \
    }

#define uct_ib_log_exp_post_send(_qp, _wr, _dump_cb) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ib_log_exp_post_send(__FILE__, __LINE__, __FUNCTION__, _qp, _wr, _dump_cb); \
    }

#endif
