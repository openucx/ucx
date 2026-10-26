/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_IPC_CFT_CUH
#define UCT_CUDA_IPC_CFT_CUH

#include <uct/api/device/uct_device_types.h>
#include <ucs/sys/device_code.h>
#include <ucs/type/status.h>

/*
 * CUDA fabric transport (CFT) device operations for cuda_ipc memory elements
 * with a valid logical endpoint id.
 *
 * Fabric operations read their source from shared memory, so data is staged
 * through a static shared memory buffer of UCT_CUDA_IPC_CFT_STAGE_SIZE bytes
 * per warp. Warp and thread level operations use the slot of the calling
 * warp, block level operations use the buffers of all warps. Block level and
 * warp or thread level operations must not run concurrently in the same block.
 * Operations complete before returning.
 */

/* Shared memory staging buffer size per warp, 0 disables CFT operations */
#ifndef UCT_CUDA_IPC_CFT_STAGE_SIZE
#define UCT_CUDA_IPC_CFT_STAGE_SIZE 256
#endif

/* Maximal number of warps in a block using CFT operations */
#ifndef UCT_CUDA_IPC_CFT_MAX_WARPS
#define UCT_CUDA_IPC_CFT_MAX_WARPS 32
#endif

#define UCT_CUDA_IPC_CFT_ALIGN 16

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    ((__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__) >= 1303) && \
    (UCT_CUDA_IPC_CFT_STAGE_SIZE > 0)
#define UCT_CUDA_IPC_CFT_ENABLE 1
#else
#define UCT_CUDA_IPC_CFT_ENABLE 0
#endif


template<ucs_device_level_t level>
UCS_F_DEVICE void uct_cuda_ipc_get_lane(unsigned &lane_id, unsigned &num_lanes);

template<ucs_device_level_t level> UCS_F_DEVICE void uct_cuda_ipc_level_sync();


#if UCT_CUDA_IPC_CFT_ENABLE

static_assert((UCT_CUDA_IPC_CFT_STAGE_SIZE % UCT_CUDA_IPC_CFT_ALIGN) == 0,
              "CFT stage size must be a multiple of 16 bytes");

typedef struct {
    alignas(UCT_CUDA_IPC_CFT_ALIGN)
    uint8_t      stage[UCT_CUDA_IPC_CFT_MAX_WARPS][UCT_CUDA_IPC_CFT_STAGE_SIZE];
    uint64_t     bar[UCT_CUDA_IPC_CFT_MAX_WARPS]; /* mbarrier per warp */
    ucs_status_t status;
} uct_cuda_ipc_cft_smem_t;


typedef struct {
    uint8_t  *buf;
    uint32_t size;
    uint32_t bar;    /* Shared memory address of the mbarrier */
    uint32_t phase;
} uct_cuda_ipc_cft_slot_t;


UCS_F_DEVICE uct_cuda_ipc_cft_smem_t *uct_cuda_ipc_cft_smem()
{
    __shared__ uct_cuda_ipc_cft_smem_t smem;
    return &smem;
}


UCS_F_DEVICE uint32_t uct_cuda_ipc_cft_smem_addr(const void *ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}


template<ucs_device_level_t level>
UCS_F_DEVICE uct_cuda_ipc_cft_slot_t uct_cuda_ipc_cft_slot()
{
    uct_cuda_ipc_cft_smem_t *smem = uct_cuda_ipc_cft_smem();
    unsigned index                = 0;
    uct_cuda_ipc_cft_slot_t slot;

    if (level == UCS_DEVICE_LEVEL_BLOCK) {
        slot.size = sizeof(smem->stage);
    } else {
        index     = threadIdx.x / UCS_DEVICE_NUM_THREADS_IN_WARP;
        slot.size = sizeof(smem->stage[0]);
    }

    slot.buf   = smem->stage[index];
    slot.bar   = uct_cuda_ipc_cft_smem_addr(&smem->bar[index]);
    slot.phase = 0;
    return slot;
}


UCS_F_DEVICE void uct_cuda_ipc_cft_bar_init(const uct_cuda_ipc_cft_slot_t &slot)
{
    asm volatile("mbarrier.init.shared.layout::v1.b64 [%0], 1;"
                 :
                 : "r"(slot.bar)
                 : "memory");
}


UCS_F_DEVICE void uct_cuda_ipc_cft_bar_inval(const uct_cuda_ipc_cft_slot_t &slot)
{
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                 :
                 : "r"(slot.bar)
                 : "memory");
}


UCS_F_DEVICE void
uct_cuda_ipc_cft_put_block(uint32_t le_id, uint64_t le_offset, const void *src,
                           uint32_t length, const uct_cuda_ipc_cft_slot_t &slot)
{
    asm volatile("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B"
                 ".mbarrier::report::fabric.relaxed.sys.b128 "
                 "[%0, %1], [%2], %3, [%4];"
                 :
                 : "r"(le_id), "l"(le_offset),
                   "r"(uct_cuda_ipc_cft_smem_addr(src)), "r"(length),
                   "r"(slot.bar)
                 : "memory");
}


UCS_F_DEVICE void
uct_cuda_ipc_cft_put_masked(uint32_t le_id, uint64_t le_offset, const void *src,
                            uint16_t mask, const uct_cuda_ipc_cft_slot_t &slot)
{
    asm volatile("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B"
                 ".mbarrier::report::fabric.cp_mask.relaxed.sys.b128 "
                 "[%0, %1], [%2], %3, [%4], %5;"
                 :
                 : "r"(le_id), "l"(le_offset),
                   "r"(uct_cuda_ipc_cft_smem_addr(src)),
                   "r"(UCT_CUDA_IPC_CFT_ALIGN), "r"(slot.bar), "h"(mask)
                 : "memory");
}


UCS_F_DEVICE void
uct_cuda_ipc_cft_red_add_u64(uint32_t le_id, uint64_t le_offset,
                             const void *src,
                             const uct_cuda_ipc_cft_slot_t &slot)
{
    asm volatile("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B"
                 ".mbarrier::report::fabric.relaxed.sys.add.u64 "
                 "[%0, %1], [%2], %3, [%4];"
                 :
                 : "r"(le_id), "l"(le_offset),
                   "r"(uct_cuda_ipc_cft_smem_addr(src)),
                   "r"(UCT_CUDA_IPC_CFT_ALIGN), "r"(slot.bar)
                 : "memory");
}


/* Submit the staged operations of @a num_blocks 16-byte blocks and wait for
 * their completion */
UCS_F_DEVICE ucs_status_t uct_cuda_ipc_cft_submit_wait(
        uct_cuda_ipc_cft_slot_t &slot, uint32_t num_blocks)
{
    int ready, has_report = 0;
    uint32_t report       = 0;

    asm volatile("fabric.submit;" ::: "memory");
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(slot.bar), "r"(num_blocks)
                 : "memory");
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0], 1;"
                 :
                 : "r"(slot.bar)
                 : "memory");

    do {
        asm volatile("{\n\t"
                     ".reg .pred p, r;\n\t"
                     ".reg .u8 s8;\n\t"
                     "mbarrier.try_wait.parity.phase_type::primary.acquire.cta"
                     ".shared::cta.b64 p|r, s8, [%3], %4;\n\t"
                     "selp.b32 %0, 1, 0, p;\n\t"
                     "@p selp.b32 %1, 1, 0, r;\n\t"
                     "@p cvt.u32.u8 %2, s8;\n\t"
                     "}"
                     : "=r"(ready), "+r"(has_report), "+r"(report)
                     : "r"(slot.bar), "r"(slot.phase)
                     : "memory");
    } while (!ready);

    slot.phase ^= 1;
    return has_report ? UCS_ERR_IO_ERROR : UCS_OK;
}


/* Write a staged chunk which starts @a shift bytes into a 16-byte aligned
 * logical endpoint block, masking partial head and tail blocks */
UCS_F_DEVICE ucs_status_t
uct_cuda_ipc_cft_put_chunk(uint32_t le_id, uint64_t le_offset, uint32_t shift,
                           uint32_t length, uct_cuda_ipc_cft_slot_t &slot)
{
    const uint32_t end    = shift + length;
    const uint32_t total  = (end + UCT_CUDA_IPC_CFT_ALIGN - 1) &
                            ~(UCT_CUDA_IPC_CFT_ALIGN - 1);
    const uint32_t tail   = end % UCT_CUDA_IPC_CFT_ALIGN;
    const uint16_t head_mask = (uint16_t)(0xffffu << shift);
    const uint16_t tail_mask = (uint16_t)((1u << tail) - 1);
    uint32_t body_start, body_end;

    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    if (total == UCT_CUDA_IPC_CFT_ALIGN) {
        uct_cuda_ipc_cft_put_masked(le_id, le_offset, slot.buf,
                                    head_mask & (tail ? tail_mask : 0xffff),
                                    slot);
        return uct_cuda_ipc_cft_submit_wait(slot, 1);
    }

    body_start = 0;
    if (shift != 0) {
        uct_cuda_ipc_cft_put_masked(le_id, le_offset, slot.buf, head_mask, slot);
        body_start = UCT_CUDA_IPC_CFT_ALIGN;
    }

    body_end = total;
    if (tail != 0) {
        body_end = total - UCT_CUDA_IPC_CFT_ALIGN;
        uct_cuda_ipc_cft_put_masked(le_id, le_offset + body_end,
                                    slot.buf + body_end, tail_mask, slot);
    }

    if (body_end > body_start) {
        uct_cuda_ipc_cft_put_block(le_id, le_offset + body_start,
                                   slot.buf + body_start,
                                   body_end - body_start, slot);
    }

    return uct_cuda_ipc_cft_submit_wait(slot, total / UCT_CUDA_IPC_CFT_ALIGN);
}


template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_cuda_ipc_cft_bcast_status(ucs_status_t status)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_WARP:
        return (ucs_status_t)__shfl_sync(0xffffffff, (int)status, 0);
    case UCS_DEVICE_LEVEL_BLOCK:
        if (threadIdx.x == 0) {
            uct_cuda_ipc_cft_smem()->status = status;
        }
        __syncthreads();
        return uct_cuda_ipc_cft_smem()->status;
    default:
        return status;
    }
}


/* Copy to the staging buffer, with 16-byte vectors when the source and the
 * destination have the same alignment */
UCS_F_DEVICE void
uct_cuda_ipc_cft_stage_copy(uint8_t *dst, const uint8_t *src, uint32_t length,
                            unsigned lane_id, unsigned num_lanes)
{
    const uintptr_t misalign = (uintptr_t)dst % UCT_CUDA_IPC_CFT_ALIGN;
    uint32_t head            = 0;
    uint32_t body            = 0;
    uint32_t i;

    if (((uintptr_t)src % UCT_CUDA_IPC_CFT_ALIGN) == misalign) {
        head = (UCT_CUDA_IPC_CFT_ALIGN - misalign) % UCT_CUDA_IPC_CFT_ALIGN;
        head = (head < length) ? head : length;
        body = (length - head) & ~(UCT_CUDA_IPC_CFT_ALIGN - 1);
        for (i = lane_id; i < (body / sizeof(int4)); i += num_lanes) {
            reinterpret_cast<int4*>(dst + head)[i] =
                    reinterpret_cast<const int4*>(src + head)[i];
        }
    }

    for (i = lane_id; i < head; i += num_lanes) {
        dst[i] = src[i];
    }

    for (i = head + body + lane_id; i < length; i += num_lanes) {
        dst[i] = src[i];
    }
}


template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t
uct_cuda_ipc_cft_put_level(uint32_t le_id, uint64_t le_offset,
                           const void *address, size_t length)
{
    uct_cuda_ipc_cft_slot_t slot = uct_cuda_ipc_cft_slot<level>();
    const uint8_t *src           = static_cast<const uint8_t*>(address);
    ucs_status_t status          = UCS_OK;
    unsigned lane_id, num_lanes;
    uint32_t shift, chunk;
    size_t done;

    uct_cuda_ipc_get_lane<level>(lane_id, num_lanes);
    if (lane_id == 0) {
        uct_cuda_ipc_cft_bar_init(slot);
    }

    for (done = 0; done < length; done += chunk) {
        shift = (le_offset + done) % UCT_CUDA_IPC_CFT_ALIGN;
        chunk = ((length - done) < (slot.size - shift)) ? (length - done) :
                                                          (slot.size - shift);

        uct_cuda_ipc_cft_stage_copy(slot.buf + shift, src + done, chunk,
                                    lane_id, num_lanes);
        uct_cuda_ipc_level_sync<level>();

        if ((lane_id == 0) && (status == UCS_OK)) {
            status = uct_cuda_ipc_cft_put_chunk(le_id, le_offset + done - shift,
                                                shift, chunk, slot);
        }
        uct_cuda_ipc_level_sync<level>();
    }

    if (lane_id == 0) {
        uct_cuda_ipc_cft_bar_inval(slot);
    }

    return uct_cuda_ipc_cft_bcast_status<level>(status);
}


UCS_F_DEVICE ucs_status_t uct_cuda_ipc_cft_atomic_add_thread(
        uint32_t le_id, uint64_t le_offset, uint64_t inc_value,
        uct_cuda_ipc_cft_slot_t &slot)
{
    const uint64_t block_offset = le_offset & ~(uint64_t)(UCT_CUDA_IPC_CFT_ALIGN - 1);
    uint64_t *payload           = reinterpret_cast<uint64_t*>(slot.buf);
    ucs_status_t status;

    /* Order previous generic and fabric writes before the increment */
    asm volatile("fence.proxy.fabric::generic.alias.release.sys;" ::: "memory");
    asm volatile("fence.proxy.fabric::fabric.alias.release.sys;" ::: "memory");

    /* Add zero to the other half of the 16-byte block */
    payload[0] = (le_offset == block_offset) ? inc_value : 0;
    payload[1] = (le_offset == block_offset) ? 0 : inc_value;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    uct_cuda_ipc_cft_bar_init(slot);
    uct_cuda_ipc_cft_red_add_u64(le_id, block_offset, payload, slot);
    status = uct_cuda_ipc_cft_submit_wait(slot, 1);
    uct_cuda_ipc_cft_bar_inval(slot);
    return status;
}


/* Run a thread level operation for each calling lane of the warp in turn,
 * because the lanes share the staging slot of the warp */
template<typename Op>
UCS_F_DEVICE ucs_status_t uct_cuda_ipc_cft_serialize_lanes(Op op)
{
    const unsigned mask = __activemask();
    const unsigned lane = threadIdx.x % UCS_DEVICE_NUM_THREADS_IN_WARP;
    ucs_status_t status = UCS_OK;
    unsigned pending;

    for (pending = mask; pending != 0; pending &= pending - 1) {
        if ((__ffs(pending) - 1) == lane) {
            status = op();
        }
        __syncwarp(mask);
    }

    return status;
}


/* Not inlined, to keep the mapped memory path of the callers small */
template<ucs_device_level_t level>
__device__ __noinline__ static ucs_status_t uct_cuda_ipc_cft_ep_put(
        const uct_cuda_ipc_md_device_mem_element_t *elem, const void *address,
        uint64_t remote_address, size_t length)
{
    const uint64_t le_offset = remote_address + elem->le_offset_delta;

    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        return uct_cuda_ipc_cft_serialize_lanes([&]() {
            return uct_cuda_ipc_cft_put_level<UCS_DEVICE_LEVEL_THREAD>(
                    elem->le_id, le_offset, address, length);
        });
    case UCS_DEVICE_LEVEL_WARP:
    case UCS_DEVICE_LEVEL_BLOCK:
        return uct_cuda_ipc_cft_put_level<level>(elem->le_id, le_offset,
                                                 address, length);
    default:
        return UCS_ERR_UNSUPPORTED;
    }
}


template<ucs_device_level_t level>
__device__ __noinline__ static ucs_status_t uct_cuda_ipc_cft_ep_atomic_add(
        const uct_cuda_ipc_md_device_mem_element_t *elem, uint64_t inc_value,
        uint64_t remote_address)
{
    const uint64_t le_offset     = remote_address + elem->le_offset_delta;
    uct_cuda_ipc_cft_slot_t slot = uct_cuda_ipc_cft_slot<level>();
    ucs_status_t status          = UCS_OK;
    unsigned lane_id, num_lanes;

    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        return uct_cuda_ipc_cft_serialize_lanes([&]() {
            return uct_cuda_ipc_cft_atomic_add_thread(elem->le_id, le_offset,
                                                      inc_value, slot);
        });
    case UCS_DEVICE_LEVEL_WARP:
    case UCS_DEVICE_LEVEL_BLOCK:
        uct_cuda_ipc_get_lane<level>(lane_id, num_lanes);
        uct_cuda_ipc_level_sync<level>();
        if (lane_id == 0) {
            status = uct_cuda_ipc_cft_atomic_add_thread(elem->le_id, le_offset,
                                                        inc_value, slot);
        }
        return uct_cuda_ipc_cft_bcast_status<level>(status);
    default:
        return UCS_ERR_UNSUPPORTED;
    }
}

#else /* UCT_CUDA_IPC_CFT_ENABLE */

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_cuda_ipc_cft_ep_put(
        const uct_cuda_ipc_md_device_mem_element_t *elem, const void *address,
        uint64_t remote_address, size_t length)
{
    return UCS_ERR_UNSUPPORTED;
}


template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_cuda_ipc_cft_ep_atomic_add(
        const uct_cuda_ipc_md_device_mem_element_t *elem, uint64_t inc_value,
        uint64_t remote_address)
{
    return UCS_ERR_UNSUPPORTED;
}

#endif /* UCT_CUDA_IPC_CFT_ENABLE */

#endif /* UCT_CUDA_IPC_CFT_CUH */
