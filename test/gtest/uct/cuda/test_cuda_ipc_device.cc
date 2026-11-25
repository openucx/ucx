/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/uct_test.h>
#include <uct/api/device/uct_device_types.h>
#include <uct/api/v2/uct_v2.h>
#include "cuda_vmm_mem_buffer.h"
#include "test_kernels_uct.h"
#include <cuda.h>
#include <common/cuda.h>
#include <algorithm>
#include <memory>
#include <unistd.h>
#include <vector>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
}

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
        return sizeof(uct_cuda_ipc_md_device_mem_element_t);
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
    EXPECT_EQ(get_mem_elem_size(),
              sizeof(uct_cuda_ipc_md_device_mem_element_t));
}

UCS_TEST_P(test_cuda_ipc_rma, get_mem_elem_pack)
{
    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
    size_t length               = 1024;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    uct_device_mem_elem_t mem_elem_host;
    void *release_handle;
    EXPECT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), sendbuf.memh(),
                                       recvbuf.rkey(), &mem_elem_host,
                                       &release_handle));
    uct_md_mem_elem_release(m_sender->md(), release_handle);
}

UCS_TEST_P(test_cuda_ipc_rma, get_device_ep)
{
    uct_device_ep_h device_ep;

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_rma, cuda_ipc)

UCS_TEST_P(test_cuda_ipc_rma_device, put_device)
{
    size_t length                   = base_length + get_offset();
    ucs_device_level_t device_level = get_device_level();
    unsigned num_threads            = get_num_threads();
    unsigned num_blocks             = get_num_blocks();

    if (device_level == UCS_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    if ((device_level == UCS_DEVICE_LEVEL_WARP) && (num_threads < 32)) {
        GTEST_SKIP() << "Warp level is not supported for less than 32 threads";
    }

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    uct_device_mem_elem_t src_elem_host;
    void *release_handle;
    ASSERT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), sendbuf.memh(),
                                       recvbuf.rkey(), &src_elem_host,
                                       &release_handle));


    uct_device_mem_elem_t *src_elem;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr*)&src_elem,
                                       sizeof(uct_device_mem_elem_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)src_elem, &src_elem_host,
                                         sizeof(uct_device_mem_elem_t)));

    uct_device_mem_elem_t *mem_elem;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr*)&mem_elem,
                                       sizeof(uct_device_mem_elem_t)));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)mem_elem, &src_elem_host,
                                         sizeof(uct_device_mem_elem_t)));

    uct_device_ep_h device_ep;
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
    ASSERT_UCS_OK(cuda_uct::launch_uct_put(device_ep, src_elem, mem_elem,
                                           sendbuf.ptr(),
                                           (uintptr_t)recvbuf.ptr(), length,
                                           device_level, num_threads,
                                           num_blocks));
    recvbuf.pattern_check(SEED1);
    cuMemFree((CUdeviceptr)src_elem);
    cuMemFree((CUdeviceptr)mem_elem);
    uct_md_mem_elem_release(m_sender->md(), release_handle);
}

UCS_TEST_P(test_cuda_ipc_rma_device, atomic_add_device)
{
    size_t             inc_value     = get_offset();
    size_t             mem_elem_size = get_mem_elem_size();
    ucs_device_level_t device_level  = get_device_level();
    unsigned           num_threads   = get_num_threads();
    unsigned           num_blocks    = get_num_blocks();
    uct_device_ep_h device_ep;
    uct_device_mem_elem_t *mem_elem;

    if (device_level == UCS_DEVICE_LEVEL_GRID) {
        GTEST_SKIP() << "Grid level is not supported";
    }

    if ((device_level == UCS_DEVICE_LEVEL_WARP) && (num_threads < 32)) {
        GTEST_SKIP() << "Warp level is not supported for less than 32 threads";
    }

    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));

    uct_device_mem_elem_t mem_elem_host;
    void *release_handle;
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc((CUdeviceptr*)&mem_elem, mem_elem_size));
    ASSERT_UCS_OK(uct_md_mem_elem_pack(m_sender->md(), nullptr, signal.rkey(),
                                       &mem_elem_host, &release_handle));
    ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)mem_elem, &mem_elem_host,
                                         mem_elem_size));

    cuda_uct::launch_uct_atomic(device_ep, mem_elem, (uint64_t)signal.ptr(),
                                inc_value, device_level, num_threads,
                                num_blocks);
    uint64_t signal_val = inc_value;
    ASSERT_EQ(mem_buffer::compare(&signal_val, signal.ptr(), sizeof(signal_val),
                                  UCS_MEMORY_TYPE_CUDA),
              1);
    cuMemFree((CUdeviceptr)mem_elem);
    uct_md_mem_elem_release(m_sender->md(), release_handle);
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_rma_device, cuda_ipc)

class test_cuda_ipc_sgl : public test_cuda_ipc_rma {
protected:
    struct sgl_arrays {
        std::vector<std::unique_ptr<mapped_buffer>> sendbufs;
        std::vector<std::unique_ptr<mapped_buffer>> recvbufs;
        std::vector<void*> buffers;
        std::vector<size_t> lengths;
        std::vector<uct_mem_h> memhs;
        std::vector<uint64_t> remote_addrs;
        std::vector<uct_rkey_t> rkeys;
    };

    struct sgl_completion {
        uct_completion_t uct;
        unsigned         done;
    };

    enum sgl_op_t {
        SGL_OP_PUT,
        SGL_OP_GET
    };

    bool is_sgl_supported(sgl_op_t op) {
        uct_iface_attr_v2_t attr = {};
        attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                          UCT_IFACE_ATTR_FIELD_MAX_PUT_SGL_ZCOPY_COUNT |
                          UCT_IFACE_ATTR_FIELD_MAX_GET_SGL_ZCOPY_COUNT;

        if (uct_iface_query_v2(m_sender->iface(), &attr) != UCS_OK) {
            return false;
        }

        if (op == SGL_OP_PUT) {
            return (attr.cap.flags & UCT_IFACE_FLAG_V2_PUT_SGL_ZCOPY) &&
                   (attr.max_put_sgl_zcopy_count > 0);
        }

        return (attr.cap.flags & UCT_IFACE_FLAG_V2_GET_SGL_ZCOPY) &&
               (attr.max_get_sgl_zcopy_count > 0);
    }

    void init_sgl(sgl_arrays &sgl, const std::vector<size_t> &sizes,
                  sgl_op_t op = SGL_OP_PUT) {
        size_t count = sizes.size();

        sgl.buffers.resize(count);
        sgl.lengths = sizes;
        sgl.memhs.resize(count);
        sgl.remote_addrs.resize(count);
        sgl.rkeys.resize(count);

        for (size_t i = 0; i < count; ++i) {
            uint64_t local_seed  = (op == SGL_OP_PUT) ? (SEED1 + i) : SEED2;
            uint64_t remote_seed = (op == SGL_OP_PUT) ? SEED2 : (SEED1 + i);

            sgl.sendbufs.emplace_back(new mapped_buffer(sizes[i], local_seed,
                                                        *m_sender, 0,
                                                        UCS_MEMORY_TYPE_CUDA));
            sgl.recvbufs.emplace_back(new mapped_buffer(sizes[i], remote_seed,
                                                        *m_receiver, 0,
                                                        UCS_MEMORY_TYPE_CUDA));
            sgl.buffers[i]      = sgl.sendbufs[i]->ptr();
            sgl.memhs[i]        = sgl.sendbufs[i]->memh();
            sgl.remote_addrs[i] = (uint64_t)sgl.recvbufs[i]->ptr();
            sgl.rkeys[i]        = sgl.recvbufs[i]->rkey();
        }
    }

    ucs_status_t sgl_op(sgl_op_t op, const sgl_arrays &sgl,
                        uct_completion_t *comp = NULL) {
        if (op == SGL_OP_PUT) {
            return uct_ep_put_sgl_zcopy(m_sender->ep(0), sgl.buffers.data(),
                                        sgl.lengths.data(), sgl.memhs.data(),
                                        sgl.remote_addrs.data(),
                                        sgl.rkeys.data(), NULL, NULL,
                                        sgl.lengths.size(), comp);
        }

        return uct_ep_get_sgl_zcopy(m_sender->ep(0), sgl.buffers.data(),
                                    sgl.lengths.data(), sgl.memhs.data(),
                                    sgl.remote_addrs.data(), sgl.rkeys.data(),
                                    NULL, NULL, sgl.lengths.size(), comp);
    }

    void check_sgl(sgl_op_t op, const sgl_arrays &sgl) {
        for (size_t i = 0; i < sgl.recvbufs.size(); ++i) {
            if (op == SGL_OP_PUT) {
                sgl.recvbufs[i]->pattern_check(SEED1 + i);
            } else {
                sgl.sendbufs[i]->pattern_check(SEED1 + i);
            }
        }
    }

    void test_sgl_various_counts(sgl_op_t op) {
        static constexpr size_t length = 2 * UCS_KBYTE;
        static const size_t counts[]   = {1, 2, 4, 10, 1024};

        for (size_t count : counts) {
            sgl_arrays sgl;
            init_sgl(sgl, std::vector<size_t>(count, length), op);
            ASSERT_UCS_OK_OR_INPROGRESS(sgl_op(op, sgl));
            m_sender->flush();
            check_sgl(op, sgl);

            if (HasFailure()) {
                break;
            }
        }
    }

    void test_sgl_various_lengths(sgl_op_t op) {
        sgl_arrays sgl;
        init_sgl(sgl, {64, 256, UCS_KBYTE, 4 * UCS_KBYTE, 16 * UCS_KBYTE}, op);
        ASSERT_UCS_OK_OR_INPROGRESS(sgl_op(op, sgl));
        m_sender->flush();
        check_sgl(op, sgl);
    }

    void test_sgl_with_callback(sgl_op_t op) {
        sgl_arrays sgl;
        init_sgl(sgl, std::vector<size_t>(10, UCS_KBYTE), op);

        sgl_completion comp = {};
        comp.uct.func       = completion_cb;
        comp.uct.count      = 1;
        comp.uct.status     = UCS_OK;

        ucs_status_t status = sgl_op(op, sgl, &comp.uct);
        ASSERT_UCS_OK_OR_INPROGRESS(status);

        if (status == UCS_INPROGRESS) {
            wait_for_flag(&comp.done);
            EXPECT_EQ(1u, comp.done);
            EXPECT_UCS_OK(comp.uct.status);
        }

        m_sender->flush();
        check_sgl(op, sgl);
    }

    static void completion_cb(uct_completion_t *self) {
        ucs_container_of(self, sgl_completion, uct)->done = 1;
    }

#if HAVE_CUDA_FABRIC
    struct remote_rkey {
        explicit remote_rkey(uct_component_h component) : component(component)
        {
        }

        ~remote_rkey()
        {
            if (unpacked) {
                EXPECT_UCS_OK(uct_rkey_release(component, &bundle));
            }
        }

        uct_component_h   component;
        uct_rkey_bundle_t bundle   = {};
        bool              unpacked = false;
    };

    /* Pack a multi-chunk key of the receiver and unpack it as another process
     * would, so that mapping it goes through the remote cache rather than the
     * same-process shortcut */
    void unpack_remote_rkey(uct_mem_h memh, void *address, size_t length,
                            remote_rkey &rkey)
    {
        std::vector<uint8_t> buffer(m_receiver->md_attr().rkey_packed_size);
        uct_md_mkey_pack_params_t pack_params  = {};
        uct_rkey_unpack_params_t unpack_params = {};
        uct_cuda_ipc_rkey_t *packed;
        ucs_status_t status;

        ASSERT_UCS_OK(uct_md_mkey_pack_v2(m_receiver->md(), memh, address,
                                          length, &pack_params,
                                          buffer.data()));
        packed = reinterpret_cast<uct_cuda_ipc_rkey_t*>(buffer.data());
        ASSERT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI,
                  packed->ph.handle_type);
        packed->pid = getpid() + 100000;

        unpack_params.field_mask = UCT_RKEY_UNPACK_FIELD_FLAGS;
        unpack_params.flags      = UCT_RKEY_UNPACK_FLAG_FREE_UNUSED_MAPPING;
        status = uct_rkey_unpack_v2(rkey.component, buffer.data(),
                                    &unpack_params, &rkey.bundle);
        if (status == UCS_ERR_UNREACHABLE) {
            UCS_TEST_SKIP_R("multi-chunk key is unreachable");
        }

        ASSERT_UCS_OK(status);
        rkey.unpacked = true;
    }

    static std::vector<uint8_t> read_device(const void *ptr, size_t length)
    {
        std::vector<uint8_t> data(length);

        EXPECT_EQ(CUDA_SUCCESS,
                  cuMemcpyDtoH(data.data(), (CUdeviceptr)ptr, length));
        return data;
    }

    /* One batch with elements inside a chunk, across chunk boundaries, and
     * through a key of a sub-range of the same registration */
    void test_sgl_multi_chunk(sgl_op_t op)
    {
        static constexpr unsigned num_chunks = 4;
        struct element {
            size_t offset;
            size_t length;
            bool   sub_range;
        };

        mem_buffer::set_device_context();

        cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
        const size_t chunk = alloc.chunk_size();
        const std::vector<element> elements = {
            {chunk / 4, 4 * UCS_KBYTE, false},
            {chunk - (2 * UCS_KBYTE), 4 * UCS_KBYTE, false},
            {(2 * chunk) - 100, chunk + 200, false},
            {(3 * chunk) + 512, UCS_KBYTE, true}
        };
        ucs::handle<uct_mem_h, uct_md_h> memh;
        std::vector<uint8_t> remote(alloc.size());
        remote_rkey rkey(GetParam()->component);
        remote_rkey sub_rkey(GetParam()->component);
        std::vector<uint8_t> expected;
        uct_mem_h uct_memh;
        sgl_arrays sgl;

        for (size_t i = 0; i < remote.size(); ++i) {
            remote[i] = (uint8_t)((i / chunk) * 64 + (i % 61));
        }
        ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)alloc.ptr(),
                                             remote.data(), remote.size()));

        ASSERT_UCS_OK(uct_md_mem_reg(m_receiver->md(), alloc.ptr(),
                                     alloc.size(), UCT_MD_MEM_ACCESS_ALL,
                                     &uct_memh));
        memh.reset(uct_memh, [](uct_mem_h handle, uct_md_h md) {
            EXPECT_UCS_OK(uct_md_mem_dereg(md, handle));
        }, m_receiver->md());

        ASSERT_NO_FATAL_FAILURE(
                unpack_remote_rkey(memh, alloc.ptr(), alloc.size(), rkey));
        ASSERT_NO_FATAL_FAILURE(unpack_remote_rkey(
                memh, UCS_PTR_BYTE_OFFSET(alloc.ptr(), 2 * chunk), 2 * chunk,
                sub_rkey));

        for (size_t i = 0; i < elements.size(); ++i) {
            sgl.sendbufs.emplace_back(new mapped_buffer(elements[i].length,
                                                        SEED1 + i, *m_sender,
                                                        0,
                                                        UCS_MEMORY_TYPE_CUDA));
            sgl.buffers.push_back(sgl.sendbufs[i]->ptr());
            sgl.lengths.push_back(elements[i].length);
            sgl.memhs.push_back(sgl.sendbufs[i]->memh());
            sgl.remote_addrs.push_back(
                    (uint64_t)alloc.ptr() + elements[i].offset);
            sgl.rkeys.push_back(elements[i].sub_range ? sub_rkey.bundle.rkey :
                                                        rkey.bundle.rkey);
        }

        expected = remote;
        if (op == SGL_OP_PUT) {
            for (size_t i = 0; i < elements.size(); ++i) {
                auto data = read_device(sgl.buffers[i], elements[i].length);
                std::copy(data.begin(), data.end(),
                          expected.begin() + elements[i].offset);
            }
        }

        ASSERT_UCS_OK_OR_INPROGRESS(sgl_op(op, sgl));
        m_sender->flush();

        if (op == SGL_OP_PUT) {
            EXPECT_EQ(expected, read_device(alloc.ptr(), alloc.size()));
        } else {
            for (size_t i = 0; i < elements.size(); ++i) {
                auto begin = remote.begin() + elements[i].offset;
                EXPECT_EQ(std::vector<uint8_t>(begin,
                                               begin + elements[i].length),
                          read_device(sgl.buffers[i], elements[i].length))
                        << "element " << i;
            }
        }
    }
#endif
};

UCS_TEST_P(test_cuda_ipc_sgl, iface_caps_v2)
{
    uct_iface_attr_v2_t attr = {};
    attr.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS |
                      UCT_IFACE_ATTR_FIELD_MAX_PUT_SGL_ZCOPY_COUNT |
                      UCT_IFACE_ATTR_FIELD_MAX_GET_SGL_ZCOPY_COUNT;

    ASSERT_UCS_OK(uct_iface_query_v2(m_sender->iface(), &attr));

    bool put_flag_set  = attr.cap.flags & UCT_IFACE_FLAG_V2_PUT_SGL_ZCOPY;
    bool put_count_set = attr.max_put_sgl_zcopy_count > 0;
    EXPECT_EQ(put_flag_set, put_count_set);

    bool get_flag_set  = attr.cap.flags & UCT_IFACE_FLAG_V2_GET_SGL_ZCOPY;
    bool get_count_set = attr.max_get_sgl_zcopy_count > 0;
    EXPECT_EQ(get_flag_set, get_count_set);
}

UCS_TEST_P(test_cuda_ipc_sgl, put_various_counts)
{
    if (!is_sgl_supported(SGL_OP_PUT)) {
        UCS_TEST_SKIP_R("put sgl zcopy is not supported");
    }

    test_sgl_various_counts(SGL_OP_PUT);
}

UCS_TEST_P(test_cuda_ipc_sgl, put_various_lengths)
{
    if (!is_sgl_supported(SGL_OP_PUT)) {
        UCS_TEST_SKIP_R("put sgl zcopy is not supported");
    }

    test_sgl_various_lengths(SGL_OP_PUT);
}

UCS_TEST_P(test_cuda_ipc_sgl, put_with_callback)
{
    if (!is_sgl_supported(SGL_OP_PUT)) {
        UCS_TEST_SKIP_R("put sgl zcopy is not supported");
    }

    test_sgl_with_callback(SGL_OP_PUT);
}

UCS_TEST_P(test_cuda_ipc_sgl, get_various_counts)
{
    if (!is_sgl_supported(SGL_OP_GET)) {
        UCS_TEST_SKIP_R("get sgl zcopy is not supported");
    }

    test_sgl_various_counts(SGL_OP_GET);
}

UCS_TEST_P(test_cuda_ipc_sgl, get_various_lengths)
{
    if (!is_sgl_supported(SGL_OP_GET)) {
        UCS_TEST_SKIP_R("get sgl zcopy is not supported");
    }

    test_sgl_various_lengths(SGL_OP_GET);
}

UCS_TEST_P(test_cuda_ipc_sgl, get_with_callback)
{
    if (!is_sgl_supported(SGL_OP_GET)) {
        UCS_TEST_SKIP_R("get sgl zcopy is not supported");
    }

    test_sgl_with_callback(SGL_OP_GET);
}

UCS_TEST_P(test_cuda_ipc_sgl, put_multi_chunk)
{
#if HAVE_CUDA_FABRIC
    if (!is_sgl_supported(SGL_OP_PUT)) {
        UCS_TEST_SKIP_R("put sgl zcopy is not supported");
    }

    test_sgl_multi_chunk(SGL_OP_PUT);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_sgl, get_multi_chunk)
{
#if HAVE_CUDA_FABRIC
    if (!is_sgl_supported(SGL_OP_GET)) {
        UCS_TEST_SKIP_R("get sgl zcopy is not supported");
    }

    test_sgl_multi_chunk(SGL_OP_GET);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_sgl, cuda_ipc)
