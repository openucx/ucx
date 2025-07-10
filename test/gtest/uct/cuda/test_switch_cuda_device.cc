/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/test_md.h>
#include <uct/test_p2p_rma.h>

extern "C" {
#include <ucs/sys/ptr_arith.h>
#include <uct/base/uct_md.h>
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>


class test_switch_cuda_device : public test_md {
protected:
    template<class T> void detect_mem_type(ucs_memory_type_t mem_type) const;

    void init() override {
        test_md::init();
        ASSERT_EQ(cudaGetDeviceCount(&m_num_devices), cudaSuccess);
    }

    int m_num_devices;
};

template<class T>
void test_switch_cuda_device::detect_mem_type(ucs_memory_type_t mem_type) const
{
    if (m_num_devices < 2) {
        UCS_TEST_SKIP_R("less than two cuda devices available");
    }

    int current_device;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    ASSERT_EQ(cudaSetDevice((current_device + 1) % m_num_devices), cudaSuccess);

    const size_t size = 16;
    T buffer(size, mem_type);

    EXPECT_EQ(cudaSetDevice(current_device), cudaSuccess);

    ucs_memory_type_t detected_mem_type;
    ASSERT_EQ(uct_md_detect_memory_type(m_md, buffer.ptr(), size,
                                        &detected_mem_type),
              UCS_OK);
    EXPECT_EQ(detected_mem_type, mem_type);
}

class cuda_vmm_mem_buffer {
public:
    cuda_vmm_mem_buffer() = default;
    cuda_vmm_mem_buffer(size_t size, ucs_memory_type_t mem_type);
    virtual ~cuda_vmm_mem_buffer();
    void *ptr() const;

protected:
    void init(size_t size, unsigned handle_type);

private:
    size_t m_size                               = 0;
    CUmemGenericAllocationHandle m_alloc_handle = 0;
    CUdeviceptr m_ptr                           = 0;
};

cuda_vmm_mem_buffer::cuda_vmm_mem_buffer(size_t size,
                                         ucs_memory_type_t mem_type)
{
    init(size, 0);
}

cuda_vmm_mem_buffer::~cuda_vmm_mem_buffer()
{
    cuMemUnmap(m_ptr, m_size);
    cuMemAddressFree(m_ptr, m_size);
    cuMemRelease(m_alloc_handle);
}

void *cuda_vmm_mem_buffer::ptr() const
{
    return (void*)m_ptr;
}

void cuda_vmm_mem_buffer::init(size_t size, unsigned handle_type)
{
    size_t granularity          = 0;
    CUmemAllocationProp prop    = {};
    CUmemAccessDesc access_desc = {};
    CUdevice device;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
        UCS_TEST_ABORT("failed to get the device handle for the current "
                       "context");
    }

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = device;
    if (handle_type != 0) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)handle_type;
    }
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
        CUDA_SUCCESS) {
        goto err;
    }

    m_size = ucs_align_up(size, granularity);
    if (cuMemCreate(&m_alloc_handle, m_size, &prop, 0) != CUDA_SUCCESS) {
        goto err;
    }

    if (cuMemAddressReserve(&m_ptr, m_size, 0, 0, 0) != CUDA_SUCCESS) {
        goto err_mem_release;
    }

    if (cuMemMap(m_ptr, m_size, 0, m_alloc_handle, 0) != CUDA_SUCCESS) {
        goto err_address_free;
    }

    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = device;
    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(m_ptr, m_size, &access_desc, 1) != CUDA_SUCCESS) {
        goto err_mem_unmap;
    }

    return;

err_mem_unmap:
    cuMemUnmap(m_ptr, m_size);
err_address_free:
    cuMemAddressFree(m_ptr, m_size);
err_mem_release:
    cuMemRelease(m_alloc_handle);
err:
    UCS_TEST_SKIP_R("failed to allocate CUDA fabric memory");
}

#if HAVE_CUDA_FABRIC
class cuda_fabric_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_fabric_mem_buffer(size_t size, ucs_memory_type_t mem_type);
};

cuda_fabric_mem_buffer::cuda_fabric_mem_buffer(size_t size,
                                               ucs_memory_type_t mem_type)
{
    init(size, CU_MEM_HANDLE_TYPE_FABRIC);
}
#endif

UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda_managed)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA_MANAGED);
}

UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda_vmm)
{
    detect_mem_type<cuda_vmm_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

#if HAVE_CUDA_FABRIC
UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda_fabric)
{
    detect_mem_type<cuda_fabric_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}
#endif

_UCT_MD_INSTANTIATE_TEST_CASE(test_switch_cuda_device, cuda_cpy);

class test_p2p_create_destroy_ctx : public uct_p2p_rma_test {
public:
    void test_xfer(send_func_t send, size_t length, unsigned flags,
                   ucs_memory_type_t mem_type) override;
};

void test_p2p_create_destroy_ctx::test_xfer(send_func_t send, size_t length,
                                            unsigned flags,
                                            ucs_memory_type_t mem_type)
{
    int num_devices;
    ASSERT_EQ(cuDeviceGetCount(&num_devices), CUDA_SUCCESS);

    if (num_devices < 1) {
        UCS_TEST_SKIP_R("no cuda devices available");
    }

    CUdevice device;
    ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);

    CUcontext ctx;
#if CUDA_VERSION >= 12050
    CUctxCreateParams ctx_create_params = {};
    ASSERT_EQ(cuCtxCreate_v4(&ctx, &ctx_create_params, 0, device), CUDA_SUCCESS);
#else
    ASSERT_EQ(cuCtxCreate(&ctx, 0, device), CUDA_SUCCESS);
#endif
    uct_p2p_rma_test::test_xfer(send, length, flags, mem_type);
    EXPECT_EQ(cuCtxDestroy(ctx), CUDA_SUCCESS);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, put_short)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_short), 1,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, get_short)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_short), 1,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, put_zcopy)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
              sender().iface_attr().cap.put.min_zcopy + 1,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, get_zcopy)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
              sender().iface_attr().cap.get.min_zcopy + 1,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

_UCT_INSTANTIATE_TEST_CASE(test_p2p_create_destroy_ctx, cuda_copy)

class test_another_thread : public test_md {
public:
    template<class T> void detect_mem_type(ucs_memory_type_t mem_type) const;
};

template<class T>
void test_another_thread::detect_mem_type(ucs_memory_type_t mem_type) const
{
    const size_t size = 16;
    T buffer(size, mem_type);

    ucs_memory_type_t detected_mem_type;
    ucs_status_t status;
    std::thread([&]() {
        status = uct_md_detect_memory_type(this->m_md, buffer.ptr(), size,
                                           &detected_mem_type);
    }).join();

    ASSERT_EQ(status, UCS_OK);
    EXPECT_EQ(detected_mem_type, mem_type);
}

UCS_TEST_P(test_another_thread, detect_mem_type_cuda)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_another_thread, detect_mem_type_cuda_managed)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA_MANAGED);
}

UCS_TEST_P(test_another_thread, detect_mem_type_cuda_vmm)
{
    detect_mem_type<cuda_vmm_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

#if HAVE_CUDA_FABRIC
UCS_TEST_P(test_another_thread, detect_mem_type_cuda_fabric)
{
    detect_mem_type<cuda_fabric_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}
#endif

_UCT_MD_INSTANTIATE_TEST_CASE(test_another_thread, cuda_cpy);

class test_mem_alloc_device : public test_switch_cuda_device {
protected:
    void init() override {
        test_switch_cuda_device::init();

        for (auto device = 0; device < m_num_devices; ++device) {
            uct_md_mem_attr_t attr       = {
                .field_mask = UCT_MD_MEM_ATTR_FIELD_SYS_DEV
            };
            static constexpr size_t size = 4096;
            void *ptr;

            ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
            ASSERT_EQ(cudaSuccess, cudaMalloc(&ptr, size));
            EXPECT_UCS_OK(uct_md_mem_query(md(), ptr, size, &attr));
            ASSERT_EQ(cudaSuccess, cudaFree(ptr));

            UCS_TEST_MESSAGE << "CUDA device " << device << " sys_dev "
                             << static_cast<int>(attr.sys_dev);
            m_sys_dev.push_back(attr.sys_dev);
        }

        ASSERT_EQ(m_num_devices, m_sys_dev.size());
    }

    ucs_status_t
    allocate(ucs_memory_type_t mem_type,
             ucs_sys_device_t sys_device = UCS_SYS_DEVICE_ID_UNKNOWN,
             size_t size = 1024)
    {
        uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
        uct_md_h md_p             = md();
        uct_mem_alloc_params_t params;

        params.field_mask = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS |
                            UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                            UCT_MEM_ALLOC_PARAM_FIELD_MDS;
        params.flags      = UCT_MD_MEM_ACCESS_ALL;
        params.mem_type   = mem_type;
        params.mds.mds    = &md_p;
        params.mds.count  = 1;
        if (sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) {
            params.field_mask |= UCT_MEM_ALLOC_PARAM_FIELD_SYS_DEVICE;
            params.sys_device  = sys_device;
        }

        return uct_mem_alloc(size, &method, 1, &params, &mem);
    }

    ucs_status_t query_sys_dev(uct_allocated_memory_t &mem,
                               ucs_sys_device_t &sys_dev)
    {
        uct_md_mem_attr_t attr = {
            .field_mask = UCT_MD_MEM_ATTR_FIELD_SYS_DEV
        };

        ucs_status_t status = uct_md_mem_query(md(), mem.address, mem.length,
                                               &attr);
        sys_dev = attr.sys_dev;
        return status;
    }

    void test_per_device_alloc(ucs_memory_type_t mem_type)
    {
        CUdevice current;

        // Ensure a valid context for each device
        for (auto device = 0; device < m_num_devices; ++device) {
            ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        }

        for (auto device = 0; device < m_num_devices; ++device) {
            ASSERT_UCS_OK(allocate(mem_type, m_sys_dev[device]));
            EXPECT_NE(CU_DEVICE_INVALID, m_sys_dev[device]);

            ucs_sys_device_t sys_device;
            EXPECT_UCS_OK(query_sys_dev(mem, sys_device));
            EXPECT_EQ(m_sys_dev[device], sys_device);
            EXPECT_EQ(cudaGetDevice(&current), cudaSuccess);
            EXPECT_EQ(m_num_devices - 1, current);
            ASSERT_UCS_OK(uct_mem_free(&mem));
        }
    }

    void test_same_device_alloc(ucs_memory_type_t mem_type,
                                bool set_sys_dev = true)
    {
        for (auto device = 0; device < m_num_devices; ++device) {
            ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
            ASSERT_UCS_OK(allocate(mem_type,
               set_sys_dev ? m_sys_dev[device] : UCS_SYS_DEVICE_ID_UNKNOWN));
            ucs_sys_device_t sys_device;
            EXPECT_UCS_OK(query_sys_dev(mem, sys_device));
            EXPECT_EQ(m_sys_dev[device], sys_device);
            ASSERT_UCS_OK(uct_mem_free(&mem));
        }
    }

    void skip_if_no_fabric(ucs_memory_type_t mem_type)
    {
#if HAVE_CUDA_FABRIC
        cuda_fabric_mem_buffer test_fabric_support(1024, mem_type);
#else
        UCS_TEST_SKIP_R("build without fabric support");
#endif
    }

private:
    std::vector<ucs_sys_device_t> m_sys_dev;

public:
    uct_allocated_memory_t mem;
};

UCS_TEST_P(test_mem_alloc_device, different_device_cuda)
{
    test_per_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, different_device_cuda_fabric,
           "CUDA_COPY_ENABLE_FABRIC=y")
{
    skip_if_no_fabric(UCS_MEMORY_TYPE_CUDA);
    test_per_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda)
{
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda_fabric,
           "CUDA_COPY_ENABLE_FABRIC=y")
{
    skip_if_no_fabric(UCS_MEMORY_TYPE_CUDA);
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda_implicit)
{
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA, false);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda_fabric_implicit,
           "CUDA_COPY_ENABLE_FABRIC=y")
{
    skip_if_no_fabric(UCS_MEMORY_TYPE_CUDA);
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA, false);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_mem_alloc_device, cuda_cpy);

class test_p2p_no_current_cuda_ctx : public uct_p2p_rma_test {
public:
    void test_xfer_on_thread(send_func_t send, size_t length, unsigned flags);
};

void test_p2p_no_current_cuda_ctx::test_xfer_on_thread(send_func_t send,
                                                       size_t length,
                                                       unsigned flags)
{
    mapped_buffer sendbuf(length, SEED1, sender());
    mapped_buffer recvbuf(length, SEED2, receiver(), 0, UCS_MEMORY_TYPE_CUDA);

    std::exception_ptr thread_exception;
    std::thread([&]() {
        try {
            blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
        } catch (...) {
            thread_exception = std::current_exception();
        }
    }).join();

    if (thread_exception) {
        std::rethrow_exception(thread_exception);
    }

    check_buf(sendbuf, recvbuf, flags);
}

UCS_TEST_P(test_p2p_no_current_cuda_ctx, put_short)
{
    test_xfer_on_thread(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
                        1, TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_P(test_p2p_no_current_cuda_ctx, get_short)
{
    test_xfer_on_thread(static_cast<send_func_t>(&uct_p2p_rma_test::get_short),
                        1, TEST_UCT_FLAG_RECV_ZCOPY);
}

UCS_TEST_P(test_p2p_no_current_cuda_ctx, put_zcopy)
{
    test_xfer_on_thread(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
                        sender().iface_attr().cap.put.min_zcopy + 1,
                        TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_P(test_p2p_no_current_cuda_ctx, get_zcopy)
{
    test_xfer_on_thread(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                        sender().iface_attr().cap.get.min_zcopy + 1,
                        TEST_UCT_FLAG_RECV_ZCOPY);
}

_UCT_INSTANTIATE_TEST_CASE(test_p2p_no_current_cuda_ctx, cuda_copy)

class test_p2p_send_on_diff_device : public uct_p2p_test {
public:
    typedef ucs_status_t (test_p2p_send_on_diff_device::*rma_send_func_t)(
                          uct_ep_h, const void*, size_t, uint64_t, uct_rkey_t);

    using rkey_mem_pair_t = std::pair<uct_rkey_bundle_t, uct_mem_h>;

    test_p2p_send_on_diff_device() : uct_p2p_test(0), m_num_devices(0) {}

    template<typename mem_src_t, typename mem_dest_t>
    void send_diff_device(const std::vector<rma_send_func_t>& send_funcs,
                          ucs_memory_type_t src_type,
                          ucs_memory_type_t dest_type)
    {
        int current_device;
        ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);

        const size_t size = UCS_MBYTE;
        mem_src_t src_buf(size, src_type);
        mem_dest_t dest_buf(size, dest_type);
        rkey_mem_pair_t rkey_dest = rkey_unpack(sender(), dest_buf.ptr(), size);

        // Set different device context and call send op
        ASSERT_EQ(cudaSetDevice((current_device + 1) % m_num_devices), cudaSuccess);

        for (const auto& send_func : send_funcs) {
            ASSERT_UCS_OK_OR_INPROGRESS(
              (this->*send_func)(sender_ep(), src_buf.ptr(), size,
                                (uint64_t)dest_buf.ptr(), rkey_dest.first.rkey));
            sender().flush();
        }

        rkey_release(sender(), rkey_dest);

        // Set back an original device to avoid test failures
        ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    }

    template<typename mem_alloc_t1, typename mem_alloc_t2>
    void test_diff_device(const std::vector<rma_send_func_t>& send_funcs,
                          ucs_memory_type_t mem_type1,
                          ucs_memory_type_t mem_type2)
    {
        send_diff_device<mem_alloc_t1, mem_alloc_t2> (send_funcs, mem_type1,
                                                      mem_type2);
        send_diff_device<mem_alloc_t2, mem_alloc_t1> (send_funcs, mem_type2,
                                                      mem_type1);
    }

    ucs_status_t uct_put_short(uct_ep_h ep, const void *send_buf, size_t size,
                               uint64_t recv_buf, uct_rkey_t rkey)
    {
        return uct_ep_put_short(ep, send_buf, size, recv_buf, rkey);
    }

    ucs_status_t uct_put_zcopy(uct_ep_h ep, const void *send_buf, size_t size,
                               uint64_t recv_buf, uct_rkey_t rkey)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, send_buf, size, NULL, 1);
        return uct_ep_put_zcopy(ep, iov, iovcnt, recv_buf, rkey, NULL);
    }

    ucs_status_t uct_get_short(uct_ep_h ep, const void *send_buf, size_t size,
                               uint64_t recv_buf, uct_rkey_t rkey)
    {
        return uct_ep_get_short(ep, (void*)send_buf, size, recv_buf, rkey);
    }

    ucs_status_t uct_get_zcopy(uct_ep_h ep, const void *send_buf, size_t size,
                               uint64_t recv_buf, uct_rkey_t rkey)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, send_buf, size, NULL, 1);
        return uct_ep_get_zcopy(ep, iov, iovcnt, recv_buf, rkey, NULL);
    }

    const std::vector<rma_send_func_t> short_funcs() const
    {
        return {&test_p2p_send_on_diff_device::uct_put_short,
                &test_p2p_send_on_diff_device::uct_get_short};
    }

    const std::vector<rma_send_func_t> zcopy_funcs() const
    {
        return {&test_p2p_send_on_diff_device::uct_put_zcopy,
                &test_p2p_send_on_diff_device::uct_get_zcopy};
    }

protected:
    void init() override
    {
        ASSERT_EQ(cudaGetDeviceCount(&m_num_devices), cudaSuccess);
        if (m_num_devices < 2) {
            UCS_TEST_SKIP_R("less than two cuda devices available");
        }
        uct_p2p_test::init();
    }

private:
    rkey_mem_pair_t rkey_unpack(entity &e, void *buf, size_t size)
    {
        uct_md_mem_reg_params_t reg_params = {};
        uct_rkey_bundle_t rkey             = {};
        uct_mem_h memh                     = NULL;

        ASSERT_UCS_OK(uct_md_mem_reg_v2(e.md(), buf, size, &reg_params, &memh));

        if (e.md_attr().rkey_packed_size == 0) {
            return {rkey, memh};
        }

        std::vector<uint8_t> packed_rkey(e.md_attr().rkey_packed_size);
        uct_md_mkey_pack_params_t pack_params = {};
        ASSERT_UCS_OK(uct_md_mkey_pack_v2(e.md(), memh, buf, size, &pack_params,
                      packed_rkey.data()));

        uct_rkey_unpack_params_t unpack_params = {};
        ASSERT_UCS_OK(uct_rkey_unpack_v2(e.md()->component, packed_rkey.data(),
                      &unpack_params, &rkey));

        return {rkey, memh};
    }

    void rkey_release(entity &e, rkey_mem_pair_t &rkey)
    {
        uct_rkey_release(e.md()->component, &rkey.first);

        uct_md_mem_dereg_params_t dereg_params;
        dereg_params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        dereg_params.memh       = rkey.second;
        ASSERT_UCS_OK(uct_md_mem_dereg_v2(e.md(), &dereg_params));
    }

    int                          m_num_devices;
};

UCS_TEST_P(test_p2p_send_on_diff_device, vmm_short)
{
    test_diff_device<cuda_vmm_mem_buffer, mem_buffer>(
                     short_funcs(), UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_HOST);
}

UCS_TEST_P(test_p2p_send_on_diff_device, vmm_zcopy)
{
    test_diff_device<cuda_vmm_mem_buffer, mem_buffer>(
                     zcopy_funcs(), UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_HOST);
}

UCS_TEST_P(test_p2p_send_on_diff_device, legacy_short)
{
    test_diff_device<mem_buffer, mem_buffer>(
                     short_funcs(), UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_HOST);
}

UCS_TEST_P(test_p2p_send_on_diff_device, legacy_zcopy)
{
    test_diff_device<mem_buffer, mem_buffer>(
                     zcopy_funcs(), UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_HOST);
}

_UCT_INSTANTIATE_TEST_CASE(test_p2p_send_on_diff_device, cuda_copy)

class test_p2p_send_on_diff_device_cuda_ipc : public test_p2p_send_on_diff_device {
};

#if HAVE_CUDA_FABRIC
UCS_TEST_P(test_p2p_send_on_diff_device_cuda_ipc, fabric_zcopy)
{
    send_diff_device<cuda_fabric_mem_buffer, cuda_fabric_mem_buffer>(
                     zcopy_funcs(), UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_CUDA);
}
#endif //HAVE_CUDA_FABRIC

_UCT_INSTANTIATE_TEST_CASE(test_p2p_send_on_diff_device_cuda_ipc, cuda_ipc)
