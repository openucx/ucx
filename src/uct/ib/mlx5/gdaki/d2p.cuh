/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_D2P_CUH_
#define UCT_D2P_CUH_

#include "d2p.h"
#include "d2p_proto.h"
#include "common.cuh"

#include <uct/api/device/uct_device_types.h>
#include <ucs/sys/device_code.h>


template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_post_desc(uct_ib_d2p_gpu_ep_t *ep,
                                               unsigned channel_id,
                                               uint8_t opcode, uint32_t length,
                                               uint32_t lkey, uint64_t laddr,
                                               uint32_t rkey, uint64_t raddr,
                                               uint64_t add, uint16_t flags)
{
    ucs_status_t status = UCS_INPROGRESS;
    unsigned lane_id, num_lanes;

    uct_dev_exec_init<level>(lane_id, num_lanes);

    if (lane_id == 0) {
        auto iface = ep->iface;
        auto ch    = iface->channels + (channel_id & iface->channel_mask);
        const long long depth = UCS_BIT(iface->log_depth);
        unsigned long long pi = READ_ONCE(*ch->pi);

        for (;;) {
            unsigned long long ci = READ_ONCE(*ch->ci);
            if (static_cast<long long>(pi - ci) >= depth) {
                status = UCS_ERR_NO_RESOURCE;
                break;
            }

            unsigned long long prev = atomicCAS(ch->pi, pi, pi + 1);
            if (prev == pi) {
                break;
            }
            pi = prev;
        }

        if (status == UCS_INPROGRESS) {
            const uint32_t slot = pi & UCS_MASK(iface->log_depth);
            auto desc = reinterpret_cast<volatile uct_ib_d2p_desc_t*>(
                                ch->queue_base) +
                        slot;

            desc->opcode         = opcode;
            desc->length         = length;
            desc->ep_idx         = ep->ep_idx;
            desc->lkey           = lkey;
            desc->laddr          = laddr;
            desc->rkey           = rkey;
            desc->raddr          = raddr;
            desc->add            = add;
            desc->flags          = flags;
            const uint32_t owner = (pi >> iface->log_depth) & 0x1;
            asm volatile("st.release.sys.global.u8 [%0], %1;" ::
                         "l"(&desc->owner), "r"(owner) : "memory");
        }
    }

    return static_cast<ucs_status_t>(
            uct_dev_bcast<level>(static_cast<int>(status), lane_id));
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_ep_put(
        uct_device_ep_h tl_ep, const uct_device_mem_elem_t *src_uct_elem,
        const uct_device_mem_elem_t *tl_mem_elem, const void *address,
        uint64_t remote_address, size_t length, unsigned channel_id,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto ep     = reinterpret_cast<uct_ib_d2p_gpu_ep_t*>(tl_ep);
    auto src_ib = reinterpret_cast<const uct_ib_md_device_mem_element_t*>(
            src_uct_elem);
    auto rem_ib = reinterpret_cast<const uct_ib_md_device_mem_element_t*>(
            tl_mem_elem);
    uint16_t desc_flags = (comp != nullptr ? UCT_IB_D2P_FLAG_CQ_UPDATE : 0) |
                          (flags & UCT_DEVICE_FLAG_NODELAY ?
                                   UCT_IB_D2P_FLAG_RING_DB :
                                   0);

    return uct_ib_d2p_post_desc<level>(ep, channel_id, UCT_IB_D2P_OP_RDMA_WRITE,
                                       length, src_ib->lkey,
                                       reinterpret_cast<uint64_t>(address),
                                       rem_ib->rkey, remote_address, 0,
                                       desc_flags);
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_ep_atomic_add(
        uct_device_ep_h tl_ep, const uct_device_mem_elem_t *tl_mem_elem,
        uint64_t inc_value, uint64_t remote_address, unsigned channel_id,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto ep     = reinterpret_cast<uct_ib_d2p_gpu_ep_t*>(tl_ep);
    auto rem_ib = reinterpret_cast<const uct_ib_md_device_mem_element_t*>(
            tl_mem_elem);
    uint16_t desc_flags = (comp != nullptr ? UCT_IB_D2P_FLAG_CQ_UPDATE : 0) |
                          (flags & UCT_DEVICE_FLAG_NODELAY ?
                                   UCT_IB_D2P_FLAG_RING_DB :
                                   0);

    return uct_ib_d2p_post_desc<level>(ep, channel_id, UCT_IB_D2P_OP_ATOMIC_ADD,
                                       sizeof(uint64_t),
                                       ep->iface->atomic_result_lkey,
                                       ep->iface->atomic_result_va,
                                       rem_ib->rkey, remote_address, inc_value,
                                       desc_flags);
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_ib_d2p_ep_check_completion(
        uct_device_ep_h tl_ep, uct_device_completion_t *comp)
{
    return UCS_OK;
}

#endif /* UCT_D2P_CUH_ */
