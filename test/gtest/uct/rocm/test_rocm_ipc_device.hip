/**
 * Copyright (c) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>
#include <uct/api/device/uct_device_types.h>
#include <uct/api/v2/uct_v2.h>
#include "test_kernels_uct.h"
#include <hip/hip_runtime.h>
#include <memory>

class test_rocm_ipc_rma : public uct_test {
protected:
    void init() {
        hipError_t res_drv;
        int rocm_id;

        uct_test::init();

        rocm_id = 0;
        res_drv = hipSetDevice(rocm_id);
        if (res_drv != hipSuccess) {
            ucs_error("hipSetDevice returned %d.", res_drv);
            return;
        }

        res_drv = hipGetDevice(&m_rocm_dev);
        if (res_drv != hipSuccess) {
            ucs_error("hipGetDevice returned %d.", res_drv);
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
        return sizeof(uct_rocm_ipc_device_mem_element_t);
    }

    entity * m_sender;
    entity * m_receiver;

    int m_rocm_dev;
    static const uint64_t SEED1     = 0xABClu;
    static const uint64_t SEED2     = 0xDEFlu;
    static const unsigned WAVEFRONT_SIZE = 64;
};

UCS_TEST_P(test_rocm_ipc_rma, has_device_ep_capability)
{
    uct_iface_attr_t iface_attr;

    ASSERT_UCS_OK(uct_iface_query(m_sender->iface(), &iface_attr));
    EXPECT_EQ(iface_attr.cap.flags & UCT_IFACE_FLAG_DEVICE_EP,
              UCT_IFACE_FLAG_DEVICE_EP);
}

class test_rocm_ipc_rma_device : public test_rocm_ipc_rma {
    protected:
    void init() {
        test_rocm_ipc_rma::init();
    }

    void cleanup() {
        test_rocm_ipc_rma::cleanup();
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
    [23:12] num_threads   (int, 0..4095)  used: 1, 64, 128, 256 (threads per block)
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
        const int num_threads[] = {1, 64, 128, 256};
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

UCS_TEST_P(test_rocm_ipc_rma, mem_elem_size)
{
    EXPECT_EQ(get_mem_elem_size(), sizeof(uct_rocm_ipc_device_mem_element_t));
}

UCS_TEST_P(test_rocm_ipc_rma, get_mem_elem_pack)
{
    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
    size_t length = 1024;
    size_t mem_elem_size = get_mem_elem_size();
    uct_device_mem_element_t *mem_elem;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_ROCM);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_ROCM);

    ASSERT_EQ(hipSuccess, hipMalloc((void **)&mem_elem, mem_elem_size));
    EXPECT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), sendbuf.memh(),
                                       recvbuf.rkey(), mem_elem));
    hipFree(mem_elem);
}

UCS_TEST_P(test_rocm_ipc_rma, get_device_ep)
{
    uct_device_ep_h device_ep;

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
}

_UCT_INSTANTIATE_TEST_CASE(test_rocm_ipc_rma, rocm_ipc)

UCS_TEST_P(test_rocm_ipc_rma_device, device_put)
{
    size_t             length        = base_length + get_offset();
    ucs_device_level_t device_level  = get_device_level();
    unsigned           num_threads   = get_num_threads();
    unsigned           num_blocks    = get_num_blocks();

    if (device_level == UCS_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    if ((device_level == UCS_DEVICE_LEVEL_WARP) && (num_threads < 64)) {
        GTEST_SKIP() << "Warp level is not supported for less than 64 threads";
    }

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_ROCM);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_ROCM);

    uct_device_local_mem_list_elem_t src_elem_host;
    ASSERT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), sendbuf.memh(),
                                       recvbuf.rkey(),
                                       &src_elem_host.uct_mem_element));

    uct_device_local_mem_list_elem_t *src_elem;
    ASSERT_EQ(hipSuccess,
              hipMalloc((void**)&src_elem,
                       sizeof(uct_device_local_mem_list_elem_t)));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(src_elem, &src_elem_host,
                       sizeof(uct_device_local_mem_list_elem_t),
                       hipMemcpyHostToDevice));

    uct_device_mem_element_t *mem_elem;
    ASSERT_EQ(hipSuccess, hipMalloc((void**)&mem_elem,
                                    sizeof(uct_device_mem_element_t)));
    ASSERT_EQ(hipSuccess, hipMemcpy(mem_elem, &src_elem_host.uct_mem_element,
                                    sizeof(uct_device_mem_element_t),
                                    hipMemcpyHostToDevice));

    uct_device_ep_h device_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
    ASSERT_UCS_OK(rocm_uct::launch_uct_put(device_ep, src_elem, mem_elem,
                                           sendbuf.ptr(),
                                           (uintptr_t)recvbuf.ptr(), length,
                                           device_level, num_threads,
                                           num_blocks));
    recvbuf.pattern_check(SEED1);
    hipFree(src_elem);
    hipFree(mem_elem);
}

_UCT_INSTANTIATE_TEST_CASE(test_rocm_ipc_rma_device, rocm_ipc)
