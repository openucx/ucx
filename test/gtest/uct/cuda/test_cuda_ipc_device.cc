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
#include "test_kernels_uct.h"
#include "cuda_vmm_mem_buffer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/cuda.h>
#include <memory>
#include <vector>

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

    void skip_unsupported_level()
    {
        if (get_device_level() == UCS_DEVICE_LEVEL_GRID) {
            UCS_TEST_SKIP_R("Grid level is not supported");
        }

        if ((get_device_level() == UCS_DEVICE_LEVEL_WARP) &&
            (get_num_threads() < WARP_SIZE)) {
            UCS_TEST_SKIP_R("Warp level requires at least 32 threads");
        }
    }

    /* Sender memory element for a remote key, copied to device memory */
    class device_elem {
    public:
        device_elem(uct_md_h md, uct_rkey_t rkey)
            : m_md(md), m_release(nullptr), m_dev_ptr(0)
        {
            create(rkey);
        }

        ~device_elem()
        {
            if (m_dev_ptr != 0) {
                cuMemFree(m_dev_ptr);
            }
            if (m_release != nullptr) {
                uct_md_mem_elem_release(m_md, m_release);
            }
        }

        const uct_device_mem_elem_t *ptr() const
        {
            return reinterpret_cast<const uct_device_mem_elem_t*>(m_dev_ptr);
        }

        uint32_t le_id() const
        {
            return reinterpret_cast<const uct_cuda_ipc_md_device_mem_element_t*>(
                           &m_host)->le_id;
        }

    private:
        void create(uct_rkey_t rkey)
        {
            ASSERT_UCS_OK(uct_md_mem_elem_pack(m_md, nullptr, rkey, &m_host,
                                               &m_release));
            ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&m_dev_ptr, sizeof(m_host)));
            ASSERT_EQ(CUDA_SUCCESS,
                      cuMemcpyHtoD(m_dev_ptr, &m_host, sizeof(m_host)));
        }

        uct_md_h              m_md;
        uct_device_mem_elem_t m_host;
        void                  *m_release;
        CUdeviceptr           m_dev_ptr;
    };

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
    size_t length = base_length + get_offset();
    uct_device_ep_h device_ep;

    skip_unsupported_level();

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    device_elem elem(m_sender->md(), recvbuf.rkey());

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
    ASSERT_UCS_OK(cuda_uct::launch_uct_put(device_ep, elem.ptr(), elem.ptr(),
                                           sendbuf.ptr(),
                                           (uintptr_t)recvbuf.ptr(), length,
                                           get_device_level(),
                                           get_num_threads(),
                                           get_num_blocks()));
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_P(test_cuda_ipc_rma_device, atomic_add_device)
{
    size_t inc_value = get_offset();
    uct_device_ep_h device_ep;

    skip_unsupported_level();

    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    device_elem elem(m_sender->md(), signal.rkey());

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
    cuda_uct::launch_uct_atomic(device_ep, elem.ptr(), (uint64_t)signal.ptr(),
                                inc_value, get_device_level(),
                                get_num_threads(), get_num_blocks());
    uint64_t signal_val = inc_value;
    ASSERT_EQ(mem_buffer::compare(&signal_val, signal.ptr(), sizeof(signal_val),
                                  UCS_MEMORY_TYPE_CUDA),
              1);
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_rma_device, cuda_ipc)


#if HAVE_CUDA_LOGICAL_ENDPOINT

/* Device operations through CUDA fabric transport (CFT): memory of another
 * device is bound to a CUDA logical endpoint and written by fabric operations
 * instead of being mapped */
class test_cuda_ipc_cft : public test_cuda_ipc_rma_device {
protected:
    static const int    SEND_DEVICE = 0;
    static const int    RECV_DEVICE = 1;
    /* Multiple of the logical endpoint bind alignment */
    static const size_t REMOTE_SIZE = 2 * UCS_MBYTE;
    static const int    GUARD       = 0xa5;
    static const size_t GUARD_SIZE  = 32;

    /* Run @a func with @a device as the current CUDA device */
    template<typename F> static void on_device(int device, F func)
    {
        int prev_device = mem_buffer::get_device();

        mem_buffer::set_device(device);
        func();
        mem_buffer::set_device(prev_device);
    }

    /* Fabric-exportable VMM memory on @a device, registered with the receiver
     * and unpacked by the sender */
    class remote_buffer {
    public:
        remote_buffer(const test_cuda_ipc_cft &test, int device)
            : m_receiver(*test.m_receiver), m_device(device), m_mem{}
        {
            m_rkey.rkey = UCT_INVALID_RKEY;
            create();
        }

        ~remote_buffer()
        {
            m_receiver.rkey_release(&m_rkey);
            m_receiver.mem_type_dereg(&m_mem);
        }

        uint8_t *ptr() const
        {
            return static_cast<uint8_t*>(m_vmm->ptr());
        }

        uct_rkey_t rkey() const
        {
            return m_rkey.rkey;
        }

        void fill(size_t offset, size_t length, int value)
        {
            on_device(m_device, [&]() {
                mem_buffer::memset(ptr() + offset, length, value,
                                   UCS_MEMORY_TYPE_CUDA);
            });
        }

        std::vector<uint8_t> read(size_t offset, size_t length) const
        {
            std::vector<uint8_t> data(length);

            on_device(m_device, [&]() {
                mem_buffer::copy_from(data.data(), ptr() + offset, length,
                                      UCS_MEMORY_TYPE_CUDA);
            });
            return data;
        }

    private:
        void create()
        {
            on_device(m_device, [&]() {
                m_vmm.reset(new cuda_fabric_mem_buffer(REMOTE_SIZE,
                                                       UCS_MEMORY_TYPE_CUDA));
            });
            fill(0, REMOTE_SIZE, GUARD);

            m_mem.address  = m_vmm->ptr();
            m_mem.length   = REMOTE_SIZE;
            m_mem.mem_type = UCS_MEMORY_TYPE_CUDA;
            m_receiver.mem_type_reg(&m_mem, UCT_MD_MEM_ACCESS_ALL);
            m_receiver.rkey_unpack(&m_mem, &m_rkey);
        }

        const entity                            &m_receiver;
        int                                     m_device;
        std::unique_ptr<cuda_fabric_mem_buffer> m_vmm;
        uct_allocated_memory_t                  m_mem;
        uct_rkey_bundle_t                       m_rkey;
    };

    void init() override
    {
        if (mem_buffer::get_device_count() <= RECV_DEVICE) {
            UCS_TEST_SKIP_R("CUDA fabric transport tests require two devices");
        }

        if (!is_cft_supported()) {
            UCS_TEST_SKIP_R("CUDA logical endpoints are not supported");
        }

        modify_config("CUDA_IPC_ENABLE_CFT", "y");
        mem_buffer::set_device(SEND_DEVICE);
        ASSERT_EQ(cudaSuccess, cudaFree(0));
        test_cuda_ipc_rma_device::init();
    }

    static bool is_cft_supported()
    {
        int supported = 0;
        CUdevice device;

        return (cuDeviceGet(&device, RECV_DEVICE) == CUDA_SUCCESS) &&
               (cuDeviceGetAttribute(
                        &supported,
                        CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
                        device) == CUDA_SUCCESS) &&
               supported;
    }

    /* Put @a length bytes to @a offset of remote memory and check the data
     * and the surrounding bytes */
    void test_put(size_t offset, size_t length)
    {
        remote_buffer remote(*this, RECV_DEVICE);
        device_elem elem(m_sender->md(), remote.rkey());
        mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                              UCS_MEMORY_TYPE_CUDA);
        uct_device_ep_h device_ep;

        ASSERT_NE(UCT_CUDA_IPC_LE_ID_INVALID, elem.le_id());
        ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
        ASSERT_UCS_OK(cuda_uct::launch_uct_put(device_ep, elem.ptr(),
                                               elem.ptr(), sendbuf.ptr(),
                                               (uintptr_t)remote.ptr() + offset,
                                               length, get_device_level(),
                                               get_num_threads(),
                                               get_num_blocks()));

        on_device(RECV_DEVICE, [&]() {
            mem_buffer::pattern_check(remote.ptr() + offset, length, SEED1,
                                      UCS_MEMORY_TYPE_CUDA);
        });

        std::vector<uint8_t> guard(GUARD_SIZE, GUARD);
        EXPECT_EQ(guard, remote.read(offset + length, GUARD_SIZE));
        if (offset >= GUARD_SIZE) {
            EXPECT_EQ(guard, remote.read(offset - GUARD_SIZE, GUARD_SIZE));
        }
    }
};

UCS_TEST_P(test_cuda_ipc_cft, put_device)
{
    skip_unsupported_level();
    test_put(get_offset(), base_length + get_offset());
}

UCS_TEST_P(test_cuda_ipc_cft, put_device_large)
{
    skip_unsupported_level();
    /* Larger than all staging buffers, not a multiple of 16 bytes */
    test_put(get_offset() + 3, 100000 + get_offset());
}

UCS_TEST_P(test_cuda_ipc_cft, atomic_add_device)
{
    skip_unsupported_level();

    /* Cover both halves of the 16-byte block written by the fabric reduction */
    const size_t offset    = (get_offset() & 1) ? sizeof(uint64_t) : 0;
    const uint64_t initial = 0x1000;
    const uint64_t inc     = get_offset() + 1;
    remote_buffer remote(*this, RECV_DEVICE);
    device_elem elem(m_sender->md(), remote.rkey());
    uct_device_ep_h device_ep;
    uint64_t values[2] = {initial, initial};

    ASSERT_NE(UCT_CUDA_IPC_LE_ID_INVALID, elem.le_id());
    on_device(RECV_DEVICE, [&]() {
        mem_buffer::copy_to(remote.ptr(), values, sizeof(values),
                            UCS_MEMORY_TYPE_CUDA);
    });

    ASSERT_UCS_OK(uct_ep_get_device_ep(m_sender->ep(0), &device_ep));
    ASSERT_UCS_OK(cuda_uct::launch_uct_atomic(device_ep, elem.ptr(),
                                              (uintptr_t)remote.ptr() + offset,
                                              inc, get_device_level(),
                                              get_num_threads(),
                                              get_num_blocks()));

    std::vector<uint8_t> result = remote.read(0, sizeof(values));
    memcpy(values, result.data(), sizeof(values));
    EXPECT_EQ(initial + inc, values[offset / sizeof(uint64_t)]);
    EXPECT_EQ(initial, values[1 - (offset / sizeof(uint64_t))]);
}

UCS_TEST_P(test_cuda_ipc_cft, put_zcopy_unsupported)
{
    if ((get_device_level() != UCS_DEVICE_LEVEL_THREAD) ||
        (get_num_threads() != 1) || (get_num_blocks() != 1) ||
        (get_offset() != 0)) {
        UCS_TEST_SKIP_R("host operation does not depend on the variant");
    }

    /* Memory bound to a logical endpoint cannot be mapped for host operations */
    remote_buffer remote(*this, RECV_DEVICE);
    mapped_buffer sendbuf(base_length, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_CUDA);
    scoped_log_handler wrap_err(wrap_errors_logger);

    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              uct_ep_put_zcopy(m_sender->ep(0), sendbuf.iov(), 1,
                               (uint64_t)remote.ptr(), remote.rkey(), NULL));
}

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_cft, cuda_ipc)

#endif /* HAVE_CUDA_LOGICAL_ENDPOINT */

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

_UCT_INSTANTIATE_TEST_CASE(test_cuda_ipc_sgl, cuda_ipc)
