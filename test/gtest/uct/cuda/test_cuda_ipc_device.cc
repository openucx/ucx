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
    ucs_device_level_t get_device_level() const {
        return static_cast<ucs_device_level_t>((GetParam()->variant >> 28) & 0xF);
    }

    int get_num_blocks() const {
        return (GetParam()->variant >> 24) & 0xF;
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
    [31:28] device_level  (uct_device_level_t, 0..15)
    [27:24] num_blocks    (int, 0..15)    used: 1, 2
    [23:12] num_threads   (int, 0..4095)  used: 1, 32, 128, 256 (threads per block)
    [11:0]  offset        (int, 0..4095)  used: 0, 1, 4, 8 (send buffer offset)
*/
        static std::vector<std::unique_ptr<resource>> storage;
        static std::vector<const resource*> out;
        if (!out.empty()) {
            return out;
        }

        std::vector<const resource*> base = uct_test::enum_resources(tl_name);
        const ucs_device_level_t levels[] = {UCS_DEVICE_LEVEL_THREAD,
                                             UCS_DEVICE_LEVEL_WARP,
                                             UCS_DEVICE_LEVEL_BLOCK,
                                             UCS_DEVICE_LEVEL_GRID};
        const int num_threads[] = {1, 32, 128, 256};
        const int num_blocks[]  = {1, 2};
        const int offsets[]     = {0, 1, 4, 8};

        const size_t total = base.size() *
                             (sizeof(levels) / sizeof(levels[0])) *
                             (sizeof(num_threads) / sizeof(num_threads[0])) *
                             (sizeof(offsets) / sizeof(offsets[0]));
        storage.reserve(total);
        out.reserve(total);

        for (const resource* r : base) {
            for (ucs_device_level_t dl : levels) {
                for (int nt : num_threads) {
                    for (int off : offsets) {
                        for (int nb: num_blocks) {
                            std::unique_ptr<resource> up(new resource(*r));
                            up->variant = ((static_cast<int>(dl) & 0xF) << 28) |
                                          ((nb & 0xF) << 24) |
                                          ((nt & 0xFFF) << 12) |
                                          (off & 0xFFF);
                            switch (dl) {
                            case UCS_DEVICE_LEVEL_THREAD:
                                up->variant_name = "thread";
                                break;
                            case UCS_DEVICE_LEVEL_WARP:
                                up->variant_name = "warp";
                                break;
                            case UCS_DEVICE_LEVEL_BLOCK:
                                up->variant_name = "block";
                                break;
                            case UCS_DEVICE_LEVEL_GRID:
                                up->variant_name = "grid";
                                break;
                            default:
                                break;
                            }
                            up->variant_name += "- nt" + std::to_string(nt) +
                                                "- nb" + std::to_string(nb) +
                                                "- offset" + std::to_string(off);
                            out.push_back(up.get());
                            storage.emplace_back(std::move(up));
                        }
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
    cuMemFree((CUdeviceptr)mem_elem);
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
    ucs_device_level_t device_level  = get_device_level();
    unsigned           num_threads   = get_num_threads();
    size_t             length        = base_length + offset;
    unsigned           num_blocks    = get_num_blocks();
    uct_device_ep_h device_ep;
    uct_device_mem_element_t *mem_elem;
    void *send_buf;

    if (device_level == UCS_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    if ((device_level == UCS_DEVICE_LEVEL_WARP) && (num_threads < 32)) {
        GTEST_SKIP() << "Warp level is not supported for less than 32 threads";
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
    cuMemFree((CUdeviceptr)mem_elem);
}

UCS_TEST_P(test_cuda_ipc_rma_device, put_multi_device)
{
    size_t             mem_elem_size = get_mem_elem_size();
    ucs_device_level_t device_level  = get_device_level();
    unsigned           num_threads   = get_num_threads();
    unsigned           num_blocks    = get_num_blocks();
    size_t             offset        = get_offset();
    const int          iovcnt        = 8;
    size_t             length        = iovcnt * (base_length + offset);
    uint64_t           signal_val    = 4;
    uct_device_ep_h device_ep;
    uct_device_mem_element_t *mem_elem;
    uint64_t *remote_addresses_dev, remote_addresses[iovcnt];
    size_t *lengths_dev, lengths[iovcnt];
    void **addresses_dev, *addresses[iovcnt];

    if (device_level == UCS_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    if ((device_level == UCS_DEVICE_LEVEL_WARP) && (num_threads < 32)) {
        GTEST_SKIP() << "Warp level is not supported for less than 32 threads";
    }

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));

    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_elem, mem_elem_size * (iovcnt + 1)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&remote_addresses_dev, iovcnt * sizeof(uint64_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&lengths_dev, iovcnt * sizeof(size_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&addresses_dev, iovcnt * sizeof(void *)));

    for (int i = 0; i < iovcnt; i++) {
        size_t iov_offset = (base_length + offset) * i;
        addresses[i] = UCS_PTR_BYTE_OFFSET(sendbuf.ptr(), iov_offset);
        remote_addresses[i] = (uint64_t)UCS_PTR_BYTE_OFFSET(recvbuf.ptr(), iov_offset);
        lengths[i] = base_length;
        ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), sendbuf.memh(),
                                                 recvbuf.rkey(),
                                                 (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(mem_elem, mem_elem_size * i)));
    }

    ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), nullptr,
                                             signal.rkey(),
                                             (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(mem_elem, mem_elem_size * iovcnt)));

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)remote_addresses_dev, remote_addresses,
                                         iovcnt * sizeof(uint64_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)lengths_dev, lengths,
                                         iovcnt * sizeof(size_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)addresses_dev, addresses,
                                         iovcnt * sizeof(void*)));

    for (int i = 0; i < iovcnt; i++) {
        mem_buffer::pattern_fill(addresses[i], base_length, SEED1, UCS_MEMORY_TYPE_CUDA);
    }

    cuda_uct::launch_uct_put_multi(device_ep, mem_elem, iovcnt + 1, addresses_dev,
                                   remote_addresses_dev, lengths_dev, 4, (uint64_t)signal.ptr(),
                                   device_level, num_threads, num_blocks);

    for (int i = 0; i < iovcnt; i++) {
        mem_buffer::pattern_check(UCS_PTR_BYTE_OFFSET(recvbuf.ptr(), (base_length + offset) * i),
                                  base_length, SEED1, UCS_MEMORY_TYPE_CUDA);
    }

    ASSERT_EQ(mem_buffer::compare(&signal_val, signal.ptr(),
                                  sizeof(signal_val), UCS_MEMORY_TYPE_CUDA), 1);

    cuMemFree((CUdeviceptr)mem_elem);
    cuMemFree((CUdeviceptr)remote_addresses_dev);
    cuMemFree((CUdeviceptr)lengths_dev);
    cuMemFree((CUdeviceptr)addresses_dev);
}

UCS_TEST_P(test_cuda_ipc_rma_device, put_multi_partial_device)
{
    size_t             mem_elem_size = get_mem_elem_size();
    ucs_device_level_t device_level  = get_device_level();
    unsigned           num_threads   = get_num_threads();
    unsigned           num_blocks    = get_num_blocks();
    size_t             offset        = get_offset();
    const int          iovcnt        = 8;
    size_t             length        = iovcnt * (base_length + offset);
    uint64_t           signal_val    = 4;
    int                counter_index = 1;
    uct_device_ep_h device_ep;
    uct_device_mem_element_t *mem_elem;
    uint64_t *remote_addresses_dev, remote_addresses[iovcnt];
    size_t *lengths_dev, lengths[iovcnt];
    void **addresses_dev, *addresses[iovcnt];
    unsigned *mem_list_indices_dev, mem_list_indices[iovcnt + 1];

    if (device_level == UCS_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    if ((device_level == UCS_DEVICE_LEVEL_WARP) && (num_threads < 32)) {
        GTEST_SKIP() << "Warp level is not supported for less than 32 threads";
    }

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));

    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_elem, mem_elem_size * (iovcnt + 1)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&remote_addresses_dev, iovcnt * sizeof(uint64_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&lengths_dev, iovcnt * sizeof(size_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&addresses_dev, iovcnt * sizeof(void *)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_list_indices_dev, (iovcnt + 1) * sizeof(unsigned)));


    for (int i = 0, j = 0; i < iovcnt + 1; i++) {
        if (i == counter_index) {
            continue;
        }
        mem_list_indices[i] = j++;
    }
    mem_list_indices[counter_index] = iovcnt;

    for (int i = 0; i < iovcnt + 1; i++) {
        uct_rkey_t rkey;
        uct_mem_h memh;
        uct_device_mem_element_t *mem_elem_iov;

        if (i == counter_index) {
            rkey = signal.rkey();
            memh = nullptr;
        } else {
            rkey = recvbuf.rkey();
            memh = sendbuf.memh();
        }

        mem_elem_iov = (uct_device_mem_element_t*)UCS_PTR_BYTE_OFFSET(mem_elem,
                                                                      mem_elem_size * mem_list_indices[i]);

        ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), memh, rkey,
                                                 mem_elem_iov));
    }

    for (int i = 0; i < iovcnt; i++) {
        size_t iov_offset = (base_length + offset) * i;
        addresses[i] = UCS_PTR_BYTE_OFFSET(sendbuf.ptr(), iov_offset);
        remote_addresses[i] = (uint64_t)UCS_PTR_BYTE_OFFSET(recvbuf.ptr(), iov_offset);
        lengths[i] = base_length;
    }

    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)remote_addresses_dev, remote_addresses,
                                         iovcnt * sizeof(uint64_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)lengths_dev, lengths,
                                         iovcnt * sizeof(size_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)addresses_dev, addresses,
                                         iovcnt * sizeof(void*)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)mem_list_indices_dev, mem_list_indices,
                                         (iovcnt + 1) * sizeof(unsigned)));
    for (int i = 0; i < iovcnt; i++) {
        mem_buffer::pattern_fill(addresses[i], base_length, SEED1, UCS_MEMORY_TYPE_CUDA);
    }

    cuda_uct::launch_uct_put_multi_partial(device_ep, mem_elem, mem_list_indices_dev,
                                           iovcnt + 1, addresses_dev,
                                           remote_addresses_dev, lengths_dev,
                                           counter_index, signal_val, (uint64_t)signal.ptr(),
                                           device_level, num_threads, num_blocks);

    for (int i = 0; i < iovcnt; i++) {
        mem_buffer::pattern_check(UCS_PTR_BYTE_OFFSET(recvbuf.ptr(), (base_length + offset) * i),
                                  base_length, SEED1, UCS_MEMORY_TYPE_CUDA);
    }

    ASSERT_EQ(mem_buffer::compare(&signal_val, signal.ptr(),
                                  sizeof(signal_val), UCS_MEMORY_TYPE_CUDA), 1);

    cuMemFree((CUdeviceptr)mem_elem);
    cuMemFree((CUdeviceptr)remote_addresses_dev);
    cuMemFree((CUdeviceptr)lengths_dev);
    cuMemFree((CUdeviceptr)addresses_dev);
}

UCS_TEST_P(test_cuda_ipc_rma_device, atomic_add_device)
{
    size_t             inc_value     = get_offset();
    size_t             mem_elem_size = get_mem_elem_size();
    ucs_device_level_t device_level  = get_device_level();
    unsigned           num_threads   = get_num_threads();
    unsigned           num_blocks    = get_num_blocks();
    uct_device_ep_h device_ep;
    uct_device_mem_element_t *mem_elem;

    if (device_level == UCS_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    if ((device_level == UCS_DEVICE_LEVEL_WARP) && (num_threads < 32)) {
        GTEST_SKIP() << "Warp level is not supported for less than 32 threads";
    }

    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));

    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr *)&mem_elem, mem_elem_size));
    ASSERT_UCS_OK(uct_iface_mem_element_pack(m_sender->iface(), nullptr,
                                             signal.rkey(), mem_elem));

    cuda_uct::launch_uct_atomic(device_ep, mem_elem, (uint64_t)signal.ptr(), inc_value,
                                device_level, num_threads, num_blocks);
    uint64_t signal_val = inc_value;
    ASSERT_EQ(mem_buffer::compare(&signal_val, signal.ptr(),
                                  sizeof(signal_val), UCS_MEMORY_TYPE_CUDA), 1);
    cuMemFree((CUdeviceptr)mem_elem);
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_rma_device, cuda_ipc)
