/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_PROTO_H_
#define UCT_D2P_PROTO_H_

#include <ucs/sys/compiler_def.h>


enum {
    UCT_IB_D2P_OP_RDMA_WRITE = 0,
    UCT_IB_D2P_OP_ATOMIC_ADD = 1,
};


enum {
    UCT_IB_D2P_FLAG_CQ_UPDATE = UCS_BIT(0),
};


typedef struct {
    uint8_t  owner;
    uint8_t  opcode;
    uint16_t flags;
    uint32_t length;
    uint64_t qp_idx;
    uint32_t lkey;
    uint32_t rkey;
    uint64_t laddr;
    uint64_t raddr;
    uint64_t add;
    uint8_t  pad[16];
} uct_ib_d2p_desc_t UCS_V_ALIGNED(64);

#endif /* UCT_D2P_PROTO_H_ */
