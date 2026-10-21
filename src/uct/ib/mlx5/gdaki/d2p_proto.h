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
    UCT_IB_D2P_FLAG_RING_DB   = UCS_BIT(1),
};


enum {
    UCT_IB_D2P_DESC_SEG_COUNT = 6,
};


typedef struct {
    union {
        uint64_t op_len_flags;
        struct {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            uint16_t opcode;
            uint16_t flags;
            uint32_t length;
#else
            uint32_t length;
            uint16_t flags;
            uint16_t opcode;
#endif
        };
    } UCS_S_PACKED;
    uint64_t ep_id;
    union {
        uint64_t lkey_rkey;
        struct {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            uint32_t rkey;
            uint32_t lkey;
#else
            uint32_t lkey;
            uint32_t rkey;
#endif
        };
    } UCS_S_PACKED;
    uint64_t laddr;
    uint64_t raddr;
    uint64_t add;
    uint8_t  pad[16];
} uct_ib_d2p_desc_t UCS_V_ALIGNED(64);

#endif /* UCT_D2P_PROTO_H_ */
