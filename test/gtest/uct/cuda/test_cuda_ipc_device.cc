/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_device.h>
#include <uct/api/v2/uct_v2.h>
#include "test_kernels_uct.h"
#include <cuda.h>
#include <memory>

class test_cuda_ipc_rma : public uct_test {
protected:
    void init() {
        CUresult res_drv;
        int cuda_id;

        uct_test::init();

        cuda_id = 0;
        res_drv = cuDeviceGet(&m_cuda_dev, cuda_id);
        if (res_drv != CUDA_SUCCESS) {
            ucs_error("cuDeviceGet returned %d.", res_drv);
            return;
        }

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_sender->connect(0, *m_receiver, 0);
    }

    void cleanup() {
        uct_test::cleanup();
    }

    size_t get_mem_elem_size() {
        uct_iface_attr_v2_t iface_attr;
        iface_attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
        ASSERT_UCS_OK(uct_iface_query_v2(m_sender->iface(), &iface_attr));
        return iface_attr.device_mem_element_size;
    }

    entity * m_sender;
    entity * m_receiver;

    CUdevice m_cuda_dev;
    static const uint64_t SEED1     = 0xABClu;
    static const uint64_t SEED2     = 0xDEFlu;
    static const unsigned WARP_SIZE = 32;
};

UCS_TEST_P(test_cuda_ipc_rma, has_device_ep_capability)
{
    uct_iface_attr_t iface_attr;

    ASSERT_UCS_OK(uct_iface_query(m_sender->iface(), &iface_attr));
    EXPECT_EQ(iface_attr.cap.flags & UCT_IFACE_FLAG_DEVICE_EP,
              UCT_IFACE_FLAG_DEVICE_EP);
}

UCS_TEST_P(test_cuda_ipc_rma, put_zcopy)
{
    size_t length = 1024;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_put_zcopy(m_sender->ep(0),
                                                 sendbuf.iov(), 1,
                                                 (uint64_t)recvbuf.ptr(),
                                                 recvbuf.rkey(), NULL));
    m_sender->flush();
    recvbuf.pattern_check(SEED1);
}

class test_cuda_ipc_rma_device : public test_cuda_ipc_rma {
    protected:
    void init() {
        test_cuda_ipc_rma::init();
    }

    void cleanup() {
        test_cuda_ipc_rma::cleanup();
    }
    uct_device_level_t get_device_level() const {
        return static_cast<uct_device_level_t>((GetParam()->variant >> 24) & 0xFF);
    }

    int get_num_threads() const {
        return (GetParam()->variant >> 12) & 0xFFF;
    }
    int get_offset() const {
        return GetParam()->variant & 0xFFF;
    }

    static const unsigned base_length = 1024;

    public:
    static std::vector<const resource*> enum_resources(const std::string& tl_name) {
/*
Parameter packing in resource.variant (uint32_t):
    [31:24] device_level  (uct_device_level_t, 0..255)
    [23:12] num_threads   (int, 0..4095)  used: 1, 32, 128, 256, number of threads in a block for kernel
    [11:0]  offset        (int, 0..4095)  used: 0, 1, 32, 64, send buffer offset
*/
        static std::vector<std::unique_ptr<resource>> storage;
        static std::vector<const resource*> out;
        if (!out.empty()) {
            return out;
        }

        std::vector<const resource*> base = uct_test::enum_resources(tl_name);
        const uct_device_level_t levels[] = {
            UCT_DEVICE_LEVEL_THREAD,
            UCT_DEVICE_LEVEL_WARP,
            UCT_DEVICE_LEVEL_BLOCK,
            UCT_DEVICE_LEVEL_GRID
        };
        const int num_threads[] = {1, 32, 128, 256, 512};
        const int offsets[]     = {0, 1, 4, 8};

        const size_t total = base.size() *
                             (sizeof(levels) / sizeof(levels[0])) *
                             (sizeof(num_threads) / sizeof(num_threads[0])) *
                             (sizeof(offsets) / sizeof(offsets[0]));
        storage.reserve(total);
        out.reserve(total);

        for (const resource* r : base) {
            for (uct_device_level_t dl : levels) {
                for (int nt : num_threads) {
                    for (int off : offsets) {
                        std::unique_ptr<resource> up(new resource(*r));
                        up->variant = ((static_cast<int>(dl) & 0xFF) << 24) |
                                      ((nt & 0xFFF) << 12) |
                                      (off & 0xFFF);
                        switch (dl) {
                        case UCT_DEVICE_LEVEL_THREAD:
                            up->variant_name = "thread";
                            break;
                        case UCT_DEVICE_LEVEL_WARP:
                            up->variant_name = "warp";
                            break;
                        case UCT_DEVICE_LEVEL_BLOCK:
                            up->variant_name = "block";
                            break;
                        case UCT_DEVICE_LEVEL_GRID:
                            up->variant_name = "grid";
                            break;
                        default:
                            break;
                        }
                        up->variant_name += "- nt" + std::to_string(nt) +
                                            "- offset" + std::to_string(off);
                        out.push_back(up.get());
                        storage.emplace_back(std::move(up));
                    }
                }
            }
        }
        return out;
    }
};

UCS_TEST_P(test_cuda_ipc_rma, mem_elem_size)
{
    EXPECT_EQ(get_mem_elem_size(), sizeof(uct_cuda_ipc_device_mem_element_t));
}

UCS_TEST_P(test_cuda_ipc_rma, get_mem_elem_pack)
{
    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
    size_t length = 1024;
    size_t mem_elem_size = get_mem_elem_size();
    uct_device_mem_element_t *mem_elem;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_elem, mem_elem_size));
    EXPECT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), sendbuf.memh(),
                                             recvbuf.rkey(), mem_elem));
    cuMemFree((CUdeviceptr)&mem_elem);
}

UCS_TEST_P(test_cuda_ipc_rma, get_device_ep)
{
    uct_device_ep_h device_ep;

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_rma, cuda_ipc)

UCS_TEST_P(test_cuda_ipc_rma_device, put_zcopy_device)
{
    size_t             offset        = get_offset();
    size_t             mem_elem_size = get_mem_elem_size();
    uct_device_level_t device_level  = get_device_level();
    unsigned           num_threads   = get_num_threads();
    size_t             length        = base_length + offset;
    unsigned           num_blocks    = 1;
    uct_device_ep_h device_ep;
    uct_device_mem_element_t *mem_elem;
    void *send_buf;

    if (device_level == UCT_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));

    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_elem, mem_elem_size));
    ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), sendbuf.memh(),
                                             recvbuf.rkey(), mem_elem));

    send_buf = UCS_PTR_BYTE_OFFSET(sendbuf.ptr(), offset);
    mem_buffer::pattern_fill(send_buf, base_length, SEED1, UCS_MEMORY_TYPE_CUDA);
    cuda_uct::launch_uct_put_single(device_ep, mem_elem, send_buf,
                                    (uint64_t)recvbuf.ptr(), base_length,
                                    device_level,
                                    num_threads, num_blocks);
    mem_buffer::pattern_check(recvbuf.ptr(), base_length, SEED1,
                              UCS_MEMORY_TYPE_CUDA);
    cuMemFree((CUdeviceptr)&mem_elem);
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_rma_device, cuda_ipc)
