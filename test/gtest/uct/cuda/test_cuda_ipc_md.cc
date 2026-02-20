/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <thread>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>

#include <uct/test_md.h>
#include <cuda.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_cache.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/sys/uid.h>
}

class test_cuda_ipc_md : public test_md {
protected:
    static uct_cuda_ipc_rkey_t
    unpack_common(uct_md_h md, int64_t uuid, CUdeviceptr ptr, size_t size)
    {
        uct_cuda_ipc_rkey_t rkey = {};
        uct_mem_h memh;
        EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, size, NULL, &memh));
        EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, size, NULL,
                                         &rkey));

        int64_t *uuid64 = (int64_t *)rkey.uuid.bytes;
        uuid64[0]       = uuid;
        uuid64[1]       = uuid;

        /* cuIpcOpenMemHandle used by cuda_ipc_cache does not allow to open
         * handle that was created by the same process */
        uct_rkey_unpack_params_t unpack_params = { 0 };
        EXPECT_EQ(UCS_ERR_UNREACHABLE,
                  uct_rkey_unpack_v2(md->component, &rkey, &unpack_params,
                                     NULL));

        uct_md_mem_dereg_params_t params;
        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        params.memh       = memh;
        EXPECT_UCS_OK(md->ops->mem_dereg(md, &params));
        return rkey;
    }

    static uct_cuda_ipc_rkey_t unpack(uct_md_h md, int64_t uuid)
    {
        CUdeviceptr ptr;
        EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, 64));
        uct_cuda_ipc_rkey_t rkey = unpack_common(md, uuid, ptr, 64);
        EXPECT_EQ(CUDA_SUCCESS, cuMemFree(ptr));
        return rkey;
    }

#if HAVE_CUDA_FABRIC
    static void alloc_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool,
                              CUstream *cu_stream, size_t size)
    {
        CUmemPoolProps pool_props = {};
        CUmemAccessDesc map_desc;
        CUdevice cu_device;

        EXPECT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_device));

        pool_props.allocType     = CU_MEM_ALLOCATION_TYPE_PINNED;
        pool_props.location.id   = (int)cu_device;
        pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        pool_props.handleTypes   = CU_MEM_HANDLE_TYPE_FABRIC;
        pool_props.maxSize       = size;
        map_desc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        map_desc.location        = pool_props.location;

        EXPECT_EQ(CUDA_SUCCESS,
                  cuStreamCreate(cu_stream, CU_STREAM_NON_BLOCKING));
        EXPECT_EQ(CUDA_SUCCESS, cuMemPoolCreate(mpool, &pool_props));
        EXPECT_EQ(CUDA_SUCCESS, cuMemPoolSetAccess(*mpool, &map_desc, 1));
        EXPECT_EQ(CUDA_SUCCESS,
                  cuMemAllocFromPoolAsync(ptr, size, *mpool, *cu_stream));
        EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(*cu_stream));
    }

    static void
    free_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool, CUstream *cu_stream)
    {
        EXPECT_EQ(CUDA_SUCCESS, cuMemFree(*ptr));
        EXPECT_EQ(CUDA_SUCCESS, cuMemPoolDestroy(*mpool));
        EXPECT_EQ(CUDA_SUCCESS, cuStreamDestroy(*cu_stream));
    }

    static ucs_status_t
    alloc_vmm_posix_fd(CUdeviceptr *ptr, CUmemGenericAllocationHandle *handle,
                       size_t *size)
    {
        CUmemAllocationProp prop = {};
        CUmemAccessDesc access_desc = {};
        CUdevice cu_device;
        size_t granularity;

        if (cuCtxGetDevice(&cu_device) != CUDA_SUCCESS) {
            return UCS_ERR_NO_DEVICE;
        }

        prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = cu_device;

        if (cuMemGetAllocationGranularity(&granularity, &prop,
                CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
            return UCS_ERR_UNSUPPORTED;
        }

        *size = ucs_align_up(*size, granularity);

        if (cuMemCreate(handle, *size, &prop, 0) != CUDA_SUCCESS) {
            return UCS_ERR_UNSUPPORTED;
        }

        if (cuMemAddressReserve(ptr, *size, granularity, 0, 0) != CUDA_SUCCESS) {
            cuMemRelease(*handle);
            return UCS_ERR_NO_MEMORY;
        }

        if (cuMemMap(*ptr, *size, 0, *handle, 0) != CUDA_SUCCESS) {
            cuMemAddressFree(*ptr, *size);
            cuMemRelease(*handle);
            return UCS_ERR_NO_MEMORY;
        }

        access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id   = cu_device;

        if (cuMemSetAccess(*ptr, *size, &access_desc, 1) != CUDA_SUCCESS) {
            cuMemUnmap(*ptr, *size);
            cuMemAddressFree(*ptr, *size);
            cuMemRelease(*handle);
            return UCS_ERR_NO_MEMORY;
        }

        return UCS_OK;
    }

    static void free_vmm(CUdeviceptr ptr, CUmemGenericAllocationHandle handle,
                         size_t size)
    {
        cuMemUnmap(ptr, size);
        cuMemAddressFree(ptr, size);
        cuMemRelease(handle);
    }

    void posix_fd_alloc_reg_pack(CUdeviceptr *ptr,
                                 CUmemGenericAllocationHandle *handle,
                                 size_t *size, uct_mem_h *memh,
                                 uct_cuda_ipc_rkey_t *rkey)
    {
        ucs_status_t status = alloc_vmm_posix_fd(ptr, handle, size);
        if (status == UCS_ERR_UNSUPPORTED) {
            UCS_TEST_SKIP_R("POSIX FD VMM allocation not supported");
        }
        ASSERT_UCS_OK(status);

        EXPECT_UCS_OK(md()->ops->mem_reg(md(), (void*)*ptr, *size, NULL, memh));
        EXPECT_UCS_OK(md()->ops->mkey_pack(md(), *memh, (void*)*ptr, *size,
                                           NULL, rkey));
        EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_POSIX_FD, rkey->ph.handle_type);
    }

    void posix_fd_dereg_free(uct_mem_h memh, CUdeviceptr ptr,
                             CUmemGenericAllocationHandle handle, size_t size)
    {
        uct_md_mem_dereg_params_t params;
        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        params.memh       = memh;
        EXPECT_UCS_OK(md()->ops->mem_dereg(md(), &params));
        free_vmm(ptr, handle, size);
    }

    static uct_cuda_ipc_rkey_t unpack_masync(uct_md_h md, int64_t uuid)
    {
        size_t size = 4 * UCS_MBYTE;
        CUdeviceptr ptr;
        CUmemoryPool mpool;
        CUstream cu_stream;

        alloc_mempool(&ptr, &mpool, &cu_stream, size);
        uct_cuda_ipc_rkey_t rkey = unpack_common(md, uuid, ptr, size);
        free_mempool(&ptr, &mpool, &cu_stream);
        return rkey;
    }
#endif

    void test_mkey_pack_on_thread(void *ptr, size_t size)
    {
       uct_md_mem_reg_params_t reg_params  = {};
       uct_rkey_bundle_t       rkey_bundle = {};
       uct_mem_h memh;
       ASSERT_UCS_OK(uct_md_mem_reg_v2(md(), ptr, size, &reg_params, &memh));

       std::exception_ptr thread_exception;
       std::thread([&]() {
           try {
               uct_md_mkey_pack_params_t pack_params = {};
               std::vector<uint8_t> rkey(md_attr().rkey_packed_size);
               ASSERT_UCS_OK(uct_md_mkey_pack_v2(md(), memh, ptr, size,
                                                 &pack_params, rkey.data()));

               // No context and sys_dev is not provided
               uct_rkey_unpack_params_t unpack_params = {};
               ucs_status_t status = uct_rkey_unpack_v2(
                                         md()->component, rkey.data(),
                                         &unpack_params, &rkey_bundle);
               ASSERT_EQ(status, UCS_ERR_UNREACHABLE);

               // No context and unknown sys_dev is provided
               unpack_params.field_mask = UCT_RKEY_UNPACK_FIELD_SYS_DEVICE;
               unpack_params.sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
               status = uct_rkey_unpack_v2(md()->component, rkey.data(),
                                           &unpack_params, &rkey_bundle);
               ASSERT_EQ(status, UCS_ERR_UNREACHABLE);

               // No context and some valid sys_dev is provided
               ucs_sys_device_t sys_dev;
               uct_cuda_base_get_sys_dev(0, &sys_dev);

               unpack_params.sys_device = sys_dev;
               status = uct_rkey_unpack_v2(md()->component, rkey.data(),
                                           &unpack_params, &rkey_bundle);
               ASSERT_TRUE((status == UCS_OK) ||
                           (status == UCS_ERR_UNREACHABLE));
               if (status == UCS_OK) {
                   uct_rkey_release(md()->component, &rkey_bundle);
               }
           } catch (...) {
               thread_exception = std::current_exception();
           }
       }).join();

       if (thread_exception) {
           std::rethrow_exception(thread_exception);
       }

       uct_md_mem_dereg_params_t dereg_params;
       dereg_params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
       dereg_params.memh       = memh;
       EXPECT_UCS_OK(uct_md_mem_dereg_v2(md(), &dereg_params));
    }
};

UCS_TEST_P(test_cuda_ipc_md, mpack_legacy)
{
    constexpr size_t size = 4096;
    ucs::handle<uct_md_h> md;
    uct_mem_h memh;
    uct_cuda_ipc_rkey_t rkey;
    CUdeviceptr ptr;

    UCS_TEST_CREATE_HANDLE(uct_md_h, md, uct_md_close, uct_md_open,
                           GetParam().component, GetParam().md_name.c_str(),
                           m_md_config);
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, size));
    EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, size, NULL, &memh));
    EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, size, NULL,
                                     &rkey));

    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_LEGACY, rkey.ph.handle_type);

    uct_md_mem_dereg_params_t params;
    params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
    params.memh       = memh;
    EXPECT_UCS_OK(md->ops->mem_dereg(md, &params));
    cuMemFree(ptr);
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_legacy)
{
    size_t size = 4 * UCS_MBYTE;
    CUdeviceptr ptr;

    EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, size));
    test_mkey_pack_on_thread((void*)ptr, size);
    EXPECT_EQ(CUDA_SUCCESS, cuMemFree(ptr));
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_mempool)
{
#if HAVE_CUDA_FABRIC
    int driver_version;
    EXPECT_EQ(CUDA_SUCCESS, cuDriverGetVersion(&driver_version));
    if (driver_version == 13000) {
        UCS_TEST_SKIP_R("in CUDA 13.0, calling cuMemPoolDestroy results in a "
                        "segmentation fault if "
                        "cuMemPoolExportToShareableHandle returned an error "
                        "before that");
    }

    size_t size = 4 * UCS_MBYTE;
    CUdeviceptr ptr;
    CUmemoryPool mpool;
    CUstream cu_stream;

    alloc_mempool(&ptr, &mpool, &cu_stream, size);
    test_mkey_pack_on_thread((void*)ptr, size);
    free_mempool(&ptr, &mpool, &cu_stream);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_posix_fd)
{
#if HAVE_CUDA_FABRIC
    size_t size = 4096;
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle handle;
    uct_mem_h memh;
    uct_cuda_ipc_rkey_t rkey = {};

    posix_fd_alloc_reg_pack(&ptr, &handle, &size, &memh, &rkey);
    posix_fd_dereg_free(memh, ptr, handle, size);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, posix_fd_system_id_mismatch)
{
#if HAVE_CUDA_FABRIC
    size_t size = 4096;
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle handle;
    uct_mem_h memh;
    uct_cuda_ipc_rkey_t rkey = {};

    posix_fd_alloc_reg_pack(&ptr, &handle, &size, &memh, &rkey);

    EXPECT_EQ(ucs_get_system_id(), rkey.ph.handle.posix_fd.system_id);

    /* Tamper system_id to simulate a different machine */
    rkey.ph.handle.posix_fd.system_id ^= 0xDEADBEEFDEADBEEFULL;

    /* Tamper UUID to bypass same-process shortcut */
    int64_t *uuid64 = (int64_t *)rkey.uuid.bytes;
    uuid64[0]       = 0xDEADLL;
    uuid64[1]       = 0xBEEFLL;

    uct_rkey_unpack_params_t unpack_params = { 0 };
    EXPECT_EQ(UCS_ERR_UNREACHABLE,
              uct_rkey_unpack_v2(md()->component, &rkey, &unpack_params,
                                 NULL));

    posix_fd_dereg_free(memh, ptr, handle, size);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mnnvl_disabled)
{
    /* Currently MNNVL is always disabled in CI */
    uct_cuda_ipc_md_t *cuda_ipc_md = ucs_derived_of(md(), uct_cuda_ipc_md_t);
    EXPECT_FALSE(cuda_ipc_md->enable_mnnvl);
}

UCS_TEST_P(test_cuda_ipc_md, posix_fd_same_node_ipc)
{
#if HAVE_CUDA_FABRIC
    size_t size = 4096;
    CUdeviceptr ptr;
    CUmemGenericAllocationHandle handle;
    uct_mem_h memh;
    uct_cuda_ipc_rkey_t rkey = {};

    ucs_status_t status = alloc_vmm_posix_fd(&ptr, &handle, &size);
    if (status == UCS_ERR_UNSUPPORTED) {
        UCS_TEST_SKIP_R("POSIX FD VMM allocation not supported");
    }
    ASSERT_UCS_OK(status);

    EXPECT_EQ(CUDA_SUCCESS, cuMemsetD8(ptr, 0xAB, size));

    EXPECT_UCS_OK(md()->ops->mem_reg(md(), (void *)ptr, size, NULL, &memh));
    EXPECT_UCS_OK(md()->ops->mkey_pack(md(), memh, (void *)ptr, size, NULL,
                                       &rkey));
    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_POSIX_FD, rkey.ph.handle_type);
    EXPECT_EQ(ucs_get_system_id(), rkey.ph.handle.posix_fd.system_id);

    /* Tamper UUID to bypass same-process shortcut and exercise the full
     * POSIX FD import path (pidfd_open/pidfd_getfd + cuMemImport) */
    int64_t *uuid64 = (int64_t *)rkey.uuid.bytes;
    uuid64[0]       = 0x1234LL;
    uuid64[1]       = 0x5678LL;

    uct_component_t *component = md()->component;

    /* Unpack on a separate thread with its own CUDA context */
    std::exception_ptr thread_exception;
    std::thread([&]() {
        try {
            CUdevice dev;
            CUcontext ctx;
            ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, 0));
#if CUDA_VERSION >= 13000
            CUctxCreateParams ctx_create_params = {};
            ASSERT_EQ(CUDA_SUCCESS,
                      cuCtxCreate(&ctx, &ctx_create_params, 0, dev));
#else
            ASSERT_EQ(CUDA_SUCCESS, cuCtxCreate(&ctx, 0, dev));
#endif

            bool pidfd_supported = false;
            int probe_pidfd      = syscall(SYS_pidfd_open, getpid(), 0);
            if (probe_pidfd >= 0) {
                int dup_fd      = syscall(SYS_pidfd_getfd, probe_pidfd,
                                          STDOUT_FILENO, 0);
                pidfd_supported = (dup_fd >= 0 || errno != ENOSYS);
                if (dup_fd >= 0) {
                    close(dup_fd);
                }
                close(probe_pidfd);
            }

            uct_rkey_unpack_params_t unpack_params = {};
            uct_rkey_bundle_t rkey_bundle           = {};
            ucs_status_t unpack_status = uct_rkey_unpack_v2(
                    component, &rkey, &unpack_params, &rkey_bundle);

            if (!pidfd_supported) {
                EXPECT_EQ(UCS_ERR_UNREACHABLE, unpack_status);
                cuCtxDestroy(ctx);
                return;
            }

            ASSERT_UCS_OK(unpack_status);

            uct_cuda_ipc_unpacked_rkey_t *unpacked =
                (uct_cuda_ipc_unpacked_rkey_t *)rkey_bundle.rkey;
            void *mapped_addr;
            ucs_status_t map_status = uct_cuda_ipc_map_memhandle(
                    &unpacked->super, dev, &mapped_addr,
                    UCS_LOG_LEVEL_ERROR);
            ASSERT_UCS_OK(map_status);

            std::vector<uint8_t> host_buf(size);
            ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(
                    host_buf.data(), (CUdeviceptr)mapped_addr, size));
            for (size_t i = 0; i < size; i++) {
                ASSERT_EQ(0xAB, host_buf[i])
                    << "Data mismatch at byte " << i;
            }

            status = uct_cuda_ipc_unmap_memhandle(unpacked->super.pid,
                                                  unpacked->super.d_bptr,
                                                  mapped_addr, dev, 0);
            EXPECT_UCS_OK(status);
            uct_rkey_release(component, &rkey_bundle);
            cuCtxDestroy(ctx);
        } catch (...) {
            thread_exception = std::current_exception();
        }
    }).join();

    if (thread_exception) {
        std::rethrow_exception(thread_exception);
    }

    posix_fd_dereg_free(memh, ptr, handle, size);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_ipc_md, cuda_ipc);
