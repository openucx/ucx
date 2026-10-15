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


typedef struct {
    uint8_t  owner;
    uint8_t  opcode;
    uint16_t flags;
    uint32_t length;
    uint64_t ep_idx;
    uint32_t lkey;
    uint32_t rkey;
    uint64_t laddr;
    uint64_t raddr;
    uint64_t add;
    uint64_t comp_id;
    uint8_t  pad[8];
} uct_ib_d2p_desc_t UCS_V_ALIGNED(64);


/*
 * An op holds a completion from posting to completion. A channel has as many
 * completions as ring entries, so at most that many ops are in flight.
 *
 * A completion id is the completion index with its generation above it: the
 * number of times it was taken, from 1. When an op completes, the completer
 * writes the generation to the done entry of the completion and returns the
 * completion to the pool. So the op has completed once done has reached its
 * generation, which stays true after the completion is reused.
 */
#define UCT_IB_D2P_COMP_INDEX_BITS 16
#define UCT_IB_D2P_COMP_INDEX_MASK UCS_MASK(UCT_IB_D2P_COMP_INDEX_BITS)

#define UCT_IB_D2P_COMP_ID(_gen, _index) \
    (((uint64_t)(_gen) << UCT_IB_D2P_COMP_INDEX_BITS) | (_index))
#define UCT_IB_D2P_COMP_ID_GEN(_id)   ((_id) >> UCT_IB_D2P_COMP_INDEX_BITS)
#define UCT_IB_D2P_COMP_ID_INDEX(_id) ((_id)&UCT_IB_D2P_COMP_INDEX_MASK)

/*
 * The completion info of a channel is in GPU memory, written only by the
 * completer. The "ci" of uct_ib_d2p_gpu_channel_t points to two arrays of
 * depth entries each, stored as one array of uct_ib_d2p_ci_t. The done and
 * avail of an entry are not related to each other:
 *
 *   done:  the done entries, indexed by completion index: the generation of
 *          the last use of the completion that completed, or 0
 *   avail: the pool of free completions: ids pointing to done entries;
 *          position p is in entry p mod depth. Position "pi" of the ring is
 *          position "pi" of the pool: the GPU takes both once the completion
 *          of that position was returned. The first depth positions hold the
 *          completions at generation 1, the n-th returned id goes to position
 *          depth + n. Every other lap has UCT_IB_D2P_AVAIL_STAMP set, to tell
 *          the entry of a position from the one of the lap before.
 *
 * The ring has at most UCS_BIT(UCT_IB_D2P_COMP_INDEX_BITS) entries.
 */
#define UCT_IB_D2P_AVAIL_STAMP UCS_BIT(63)

typedef struct {
    uint64_t done;
    uint64_t avail;
} uct_ib_d2p_ci_t;

#endif /* UCT_D2P_PROTO_H_ */
