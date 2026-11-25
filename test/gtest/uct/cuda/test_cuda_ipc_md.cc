/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_vmm_mem_buffer.h"

#include <cuda_runtime.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <uct/test_md.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_cache.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_vmm_multi.h>
#include <uct/cuda/cuda_ipc/cuda_ipc.inl>
#include <uct/cuda/base/cuda_iface.h>
#include <uct/cuda/base/cuda_nvml.h>
#include <ucs/sys/uid.h>
#include <uct/cuda/base/cuda_util.h>
#include <ucs/datastruct/pgtable.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/sys/sys.h>
}

class test_cuda_ipc_md : public test_md {
protected:
    static uct_cuda_ipc_extended_rkey_t
    unpack_common(uct_md_h md, int64_t uuid, CUdeviceptr ptr, size_t size)
    {
        uct_cuda_ipc_extended_rkey_t rkey;
        uct_mem_h memh;
        EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, size, NULL, &memh));
        EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, size, NULL,
                                         &rkey));

        auto uuid64     = reinterpret_cast<int64_t*>(rkey.super.uuid.bytes);
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

    static uct_cuda_ipc_extended_rkey_t unpack(uct_md_h md, int64_t uuid)
    {
        CUdeviceptr ptr;
        EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, 64));
        const uct_cuda_ipc_extended_rkey_t rkey = unpack_common(md, uuid, ptr,
                                                                64);
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

    static uct_cuda_ipc_extended_rkey_t unpack_masync(uct_md_h md, int64_t uuid)
    {
        size_t size = 4 * UCS_MBYTE;
        CUdeviceptr ptr;
        CUmemoryPool mpool;
        CUstream cu_stream;

        alloc_mempool(&ptr, &mpool, &cu_stream, size);
        const uct_cuda_ipc_extended_rkey_t rkey = unpack_common(md, uuid, ptr,
                                                                size);
        free_mempool(&ptr, &mpool, &cu_stream);
        return rkey;
    }

    /* Pack [address, address + length), a range spanning allocations */
    void vmm_multi_publish(uct_mem_h memh, void *address, size_t length,
                           uct_cuda_ipc_vmm_multi_range_t *range)
    {
        uct_cuda_ipc_memh_t *cuda_memh = static_cast<uct_cuda_ipc_memh_t*>(
                memh);
        const uct_cuda_ipc_vmm_multi_meta_t *meta;

        ASSERT_UCS_OK(uct_cuda_ipc_mkey_pack_vmm_multi_chunk(
                cuda_memh, address, length, &meta, range));
        EXPECT_EQ(cuda_memh->vmm_multi, meta);
    }

    static const uct_cuda_ipc_vmm_multi_meta_t *vmm_multi_meta(uct_mem_h memh)
    {
        return static_cast<uct_cuda_ipc_memh_t*>(memh)->vmm_multi;
    }

    /* A key for [address, address + length), filled as the pack path fills
     * it */
    void vmm_multi_key(uct_mem_h memh, void *address, size_t length,
                       uct_cuda_ipc_rkey_t *key)
    {
        const uct_cuda_ipc_vmm_multi_meta_t *meta;
        uct_cuda_ipc_vmm_multi_range_t range;

        ASSERT_NO_FATAL_FAILURE(
                vmm_multi_publish(memh, address, length, &range));
        meta = vmm_multi_meta(memh);

        *key                         = {};
        key->ph.handle_type          = UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI;
        key->ph.handle.fabric_handle = meta->fabric_handle;
        key->ph.vmm_multi            = range.info;
        key->ph.buffer_id            = meta->buffer_id;
        key->d_bptr                  = range.d_bptr;
        key->b_len                   = range.b_len;
    }

    void mem_dereg(uct_mem_h memh)
    {
        uct_md_mem_dereg_params_t params;

        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        params.memh       = memh;
        EXPECT_UCS_OK(md()->ops->mem_dereg(md(), &params));
    }

    /* The key as another process receives it, so that mapping it goes through
     * the remote cache rather than the same-process shortcut */
    static uct_cuda_ipc_extended_rkey_t
    vmm_multi_remote_key(const uct_cuda_ipc_rkey_t &key)
    {
        uct_cuda_ipc_extended_rkey_t rkey = {};

        rkey.super     = key;
        rkey.super.pid = getpid() + 100000;
        rkey.pid_ns    = ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID);
        return rkey;
    }

    static void vmm_multi_map(uct_cuda_ipc_extended_rkey_t *rkey,
                              CUdevice cu_dev, void **mapped_p,
                              uct_cuda_ipc_cache_region_t **region_p)
    {
        ASSERT_UCS_OK(uct_cuda_ipc_map_memhandle(rkey, cu_dev, mapped_p,
                                                 region_p,
                                                 UCS_LOG_LEVEL_ERROR));
        ASSERT_TRUE(*region_p != NULL);
    }

    static void vmm_multi_unmap(const uct_cuda_ipc_extended_rkey_t *rkey,
                                CUdevice cu_dev, void *mapped,
                                uct_cuda_ipc_cache_region_t *region,
                                int cache_enabled)
    {
        uct_cuda_ipc_mapping_t mapping;

        uct_cuda_ipc_mapping_init(&mapping, rkey, mapped, region);
        uct_cuda_ipc_unmap_memhandle(&mapping, cu_dev, cache_enabled);
    }

    static bool vmm_is_mapped(void *ptr)
    {
        CUmemGenericAllocationHandle handle;

        if (cuMemRetainAllocationHandle(&handle, ptr) != CUDA_SUCCESS) {
            return false;
        }

        EXPECT_EQ(CUDA_SUCCESS, cuMemRelease(handle));
        return true;
    }

    static std::vector<uint8_t> read_device(const void *ptr, size_t length)
    {
        std::vector<uint8_t> data(length);

        EXPECT_EQ(CUDA_SUCCESS,
                  cuMemcpyDtoH(data.data(), (CUdeviceptr)ptr, length));
        return data;
    }

    /* Fill the buffer with bytes tagged by their chunk, so that a chunk mapped
     * into the wrong slot shows up as a mismatch */
    static std::vector<uint8_t> vmm_multi_fill(const cuda_vmm_mem_buffer &alloc)
    {
        std::vector<uint8_t> data(alloc.size());

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = (uint8_t)((i / alloc.chunk_size()) * 64 + (i % 61));
        }

        EXPECT_EQ(CUDA_SUCCESS, cuMemcpyHtoD((CUdeviceptr)alloc.ptr(),
                                             data.data(), data.size()));
        return data;
    }

    static std::vector<uint8_t>
    vmm_multi_expected(const std::vector<uint8_t> &data,
                       const cuda_vmm_mem_buffer &alloc,
                       const uct_cuda_ipc_rkey_t &key)
    {
        size_t offset = key.d_bptr - (CUdeviceptr)alloc.ptr();

        return std::vector<uint8_t>(data.begin() + offset,
                                    data.begin() + offset + key.b_len);
    }
#endif

    void test_mkey_pack_on_thread(void *ptr, size_t size)
    {
       uct_md_mem_reg_params_t reg_params  = {};
       uct_rkey_bundle_t       rkey_bundle = {};
       uct_mem_h memh;

       reg_params.field_mask = UCT_MD_MEM_REG_FIELD_MEM_TYPE;
       reg_params.mem_type   = UCS_MEMORY_TYPE_CUDA;
       ASSERT_UCS_OK(uct_md_mem_reg_v2(md(), ptr, size, &reg_params, &memh));

       std::exception_ptr thread_exception;
       std::thread([&]() {
           try {
               uct_md_mkey_pack_params_t pack_params = {};
               std::vector<uint8_t> rkey(md_attr().rkey_packed_size);
               ASSERT_UCS_OK(uct_md_mkey_pack_v2(md(), memh, ptr, size,
                                                 &pack_params, rkey.data()));

               auto unpack_rkey = [&](const uct_rkey_unpack_params_t &unpack_params) {
                    ucs_status_t status = uct_rkey_unpack_v2(
                                             md()->component, rkey.data(),
                                             &unpack_params, &rkey_bundle);
                    ASSERT_TRUE((status == UCS_OK) ||
                                (status == UCS_ERR_UNREACHABLE));
                    if (status == UCS_OK) {
                        uct_rkey_release(md()->component, &rkey_bundle);
                    }
               };

               // No context and sys_dev is not provided
               // Reachable, because active CUDA context exists in main thread
               uct_rkey_unpack_params_t unpack_params = {};
               unpack_rkey(unpack_params);

               // No context and unknown sys_dev is provided
               // Reachable, because active CUDA context exists for some valid GPU
               unpack_params.field_mask = UCT_RKEY_UNPACK_FIELD_SYS_DEVICE;
               unpack_params.sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
               unpack_rkey(unpack_params);

               // No context and some valid sys_dev is provided
               ucs_sys_device_t sys_dev = uct_cuda_get_sys_dev(0);
               unpack_params.sys_device = sys_dev;
               unpack_rkey(unpack_params);
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

#if HAVE_DECL_SYS_PIDFD_GETFD
    ucs_status_t pack_posix_fd_key(cuda_posix_fd_mem_buffer &buf, size_t size,
                                   uct_mem_h *memh, uct_cuda_ipc_rkey_t *rkey)
    {
        uct_md_mem_reg_params_t reg_params    = {};
        uct_md_mkey_pack_params_t pack_params = {};
        ucs_status_t status;

        status = uct_md_mem_reg_v2(md(), buf.ptr(), size, &reg_params, memh);
        if (status != UCS_OK) {
            return status;
        }

        return uct_md_mkey_pack_v2(md(), *memh, buf.ptr(), size, &pack_params,
                                   rkey);
    }

    static void
    tamper_rkey_uuid(uct_cuda_ipc_rkey_t *rkey, int64_t value0, int64_t value1)
    {
        int64_t *uuid64 = reinterpret_cast<int64_t*>(rkey->uuid.bytes);
        uuid64[0]       = value0;
        uuid64[1]       = value1;
    }

#endif

    /* Open another MD with the given NVLink domain and MNNVL configuration */
    ucs_status_t open_md(const std::string &nvlink_domain,
                         const std::string &enable_mnnvl,
                         ucs::handle<uct_md_h> &md_h)
    {
        uct_md_config_t *config;
        ucs_status_t status;
        uct_md_h md;

        status = uct_md_config_read(GetParam().component, NULL, NULL, &config);
        if (status != UCS_OK) {
            return status;
        }

        EXPECT_UCS_OK(uct_config_modify(config, "CUDA_IPC_NVLINK_DOMAIN",
                                        nvlink_domain.c_str()));
        EXPECT_UCS_OK(uct_config_modify(config, "CUDA_IPC_ENABLE_MNNVL",
                                        enable_mnnvl.c_str()));

        status = uct_md_open(GetParam().component, GetParam().md_name.c_str(),
                             config, &md);
        uct_config_release(config);
        if (status == UCS_OK) {
            md_h.reset(md, uct_md_close);
        }

        return status;
    }

    static uct_cuda_ipc_md_t *ipc_md(const ucs::handle<uct_md_h> &md_h)
    {
        return ucs_derived_of(md_h.get(), uct_cuda_ipc_md_t);
    }
};

UCS_TEST_P(test_cuda_ipc_md, mpack_legacy)
{
    constexpr size_t size = 4096;
    ucs::handle<uct_md_h> md;
    uct_mem_h memh;
    uct_cuda_ipc_extended_rkey_t rkey;
    CUdeviceptr ptr;

    UCS_TEST_CREATE_HANDLE(uct_md_h, md, uct_md_close, uct_md_open,
                           GetParam().component, GetParam().md_name.c_str(),
                           m_md_config);
    ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, size));
    EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, size, NULL, &memh));
    EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, size, NULL,
                                     &rkey));

    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_LEGACY, rkey.super.ph.handle_type);

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
#if HAVE_DECL_SYS_PIDFD_GETFD
    cuda_posix_fd_mem_buffer buf(4096, UCS_MEMORY_TYPE_CUDA);
    uct_mem_h memh;
    uct_cuda_ipc_rkey_t rkey = {};

    ASSERT_UCS_OK(pack_posix_fd_key(buf, buf.size(), &memh, &rkey));
    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_POSIX_FD, rkey.ph.handle_type);
    EXPECT_EQ(ucs_get_system_id(), rkey.ph.handle.posix_fd.system_id);

    uct_md_mem_dereg_params_t dereg_params;
    dereg_params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
    dereg_params.memh       = memh;
    EXPECT_UCS_OK(uct_md_mem_dereg_v2(md(), &dereg_params));
#else
    UCS_TEST_SKIP_R("built without pidfd support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, posix_fd_system_id_mismatch)
{
#if HAVE_DECL_SYS_PIDFD_GETFD
    cuda_posix_fd_mem_buffer buf(4096, UCS_MEMORY_TYPE_CUDA);
    uct_mem_h memh;
    uct_cuda_ipc_rkey_t rkey = {};

    ASSERT_UCS_OK(pack_posix_fd_key(buf, buf.size(), &memh, &rkey));
    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_POSIX_FD, rkey.ph.handle_type);
    EXPECT_EQ(ucs_get_system_id(), rkey.ph.handle.posix_fd.system_id);

    /* Tamper system_id to simulate a different machine */
    rkey.ph.handle.posix_fd.system_id ^= 0xDEADBEEFDEADBEEFULL;

    /* Tamper UUID to bypass same-process shortcut */
    tamper_rkey_uuid(&rkey, 0xDEADLL, 0xBEEFLL);

    uct_rkey_unpack_params_t unpack_params = {0};
    EXPECT_EQ(UCS_ERR_UNREACHABLE,
              uct_rkey_unpack_v2(md()->component, &rkey, &unpack_params, NULL));

    uct_md_mem_dereg_params_t dereg_params;
    dereg_params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
    dereg_params.memh       = memh;
    EXPECT_UCS_OK(uct_md_mem_dereg_v2(md(), &dereg_params));
#else
    UCS_TEST_SKIP_R("built without pidfd support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mnnvl_detection)
{
    uct_cuda_ipc_md_t *cuda_ipc_md = ucs_derived_of(md(), uct_cuda_ipc_md_t);
    bool expected                  = false;
#if HAVE_NVML_FABRIC_INFO
    nvmlGpuFabricInfoV_t fabric_info;
    nvmlDevice_t device;
    uint64_t cluster_uuid[2] = {};

    fabric_info.version = nvmlGpuFabricInfo_v2;
    if ((UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, 0, &device) ==
         UCS_OK) &&
        (UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetGpuFabricInfoV, device,
                                 &fabric_info) == UCS_OK)) {
        /* MNNVL is enabled only if GPU 0 completed joining an NVLink fabric */
        memcpy(cluster_uuid, fabric_info.clusterUuid, sizeof(cluster_uuid));
        expected = (fabric_info.state == NVML_GPU_FABRIC_STATE_COMPLETED) &&
                   (fabric_info.status == NVML_SUCCESS) &&
                   ((cluster_uuid[0] | cluster_uuid[1]) != 0);
    }
#endif

    ASSERT_EQ(expected, !!cuda_ipc_md->enable_mnnvl);
#if HAVE_NVML_FABRIC_INFO
    if (expected) {
        EXPECT_EQ(cluster_uuid[0] ^ cluster_uuid[1] ^ fabric_info.cliqueId,
                  cuda_ipc_md->nvlink_domain);
    }
#endif
    EXPECT_EQ(ucs_get_system_id(), cuda_ipc_md->system_uuid);
}

/* The NVLink domain knob enables MNNVL and fabric support without a GPU fabric,
 * and different emulated hosts get different system UUIDs */
UCS_TEST_P(test_cuda_ipc_md, nvlink_domain_emulation)
{
    ucs::handle<uct_md_h> md_5, md_5_0, md_5_5, md_5_7, md_6_7, md_max_7;

    ASSERT_UCS_OK(open_md("5", "try", md_5));
    ASSERT_UCS_OK(open_md("5:0", "try", md_5_0));
    ASSERT_UCS_OK(open_md("5:5", "try", md_5_5));
    ASSERT_UCS_OK(open_md("5:7", "try", md_5_7));
    ASSERT_UCS_OK(open_md("0x6:7", "try", md_6_7));
    ASSERT_UCS_OK(open_md("18446744073709551615:7", "try", md_max_7));

    EXPECT_TRUE(ipc_md(md_5)->enable_mnnvl);
    EXPECT_TRUE(ipc_md(md_5_7)->enable_mnnvl);
    EXPECT_TRUE(uct_cuda_ipc_md_fabric_supported(ipc_md(md_5)));
    EXPECT_TRUE(uct_cuda_ipc_md_fabric_supported(ipc_md(md_5_7)));
    EXPECT_EQ(5u, ipc_md(md_5)->nvlink_domain);
    EXPECT_EQ(5u, ipc_md(md_5_7)->nvlink_domain);
    EXPECT_EQ(6u, ipc_md(md_6_7)->nvlink_domain);
    EXPECT_EQ(ULONG_MAX, ipc_md(md_max_7)->nvlink_domain);

    /* <host> defaults to <domain> */
    EXPECT_EQ(ipc_md(md_5)->system_uuid, ipc_md(md_5_5)->system_uuid);
    /* The host is independent of the domain */
    EXPECT_EQ(ipc_md(md_5_7)->system_uuid, ipc_md(md_6_7)->system_uuid);
    EXPECT_EQ(ipc_md(md_5_7)->system_uuid, ipc_md(md_max_7)->system_uuid);
    EXPECT_NE(ipc_md(md_5)->system_uuid, ipc_md(md_5_7)->system_uuid);
    EXPECT_NE(ucs_get_system_id(), ipc_md(md_5_7)->system_uuid);
    EXPECT_NE(ucs_get_system_id(), ipc_md(md_5_0)->system_uuid);
}

/* Requiring MNNVL does not fail when it is emulated without a GPU fabric */
UCS_TEST_P(test_cuda_ipc_md, nvlink_domain_mnnvl_required)
{
    ucs::handle<uct_md_h> md_h;

    ASSERT_UCS_OK(open_md("5:7", "y", md_h));
    EXPECT_TRUE(ipc_md(md_h)->enable_mnnvl);
    EXPECT_EQ(5u, ipc_md(md_h)->nvlink_domain);
}

/* Disabling MNNVL overrides the NVLink domain knob */
UCS_TEST_P(test_cuda_ipc_md, nvlink_domain_mnnvl_disabled)
{
    ucs::handle<uct_md_h> md_h;

    ASSERT_UCS_OK(open_md("5:7", "n", md_h));
    EXPECT_FALSE(ipc_md(md_h)->enable_mnnvl);
    EXPECT_EQ(ucs_get_system_id(), ipc_md(md_h)->system_uuid);
}

UCS_TEST_P(test_cuda_ipc_md, nvlink_domain_invalid)
{
    static const char *values[] = {"", "abc", "5:", "5:x", "5;7", "5:7:9",
                                   ":7", "-1", "5:-1", "+5", "0x", "5:0x",
                                   " 5", "5: 7", "99999999999999999999",
                                   "18446744073709551615",
                                   "5:18446744073709551615"};
    ucs::handle<uct_md_h> md_h;

    for (const char *value : values) {
        scoped_log_handler wrap_err(wrap_errors_logger);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, open_md(value, "try", md_h))
                << "'" << value << "'";
    }
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_publish_chunks)
{
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 4;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    std::vector<uct_cuda_ipc_vmm_chunk_desc_t> descs(num_chunks);
    uct_cuda_ipc_vmm_multi_range_t range, sub_range;
    const uct_cuda_ipc_vmm_multi_meta_t *meta;
    unsigned long long buffer_id;
    uct_mem_h memh;
    CUdeviceptr chunk;

    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));

    /* A range starting and ending inside allocations covers them whole */
    ASSERT_NO_FATAL_FAILURE(vmm_multi_publish(
            memh, UCS_PTR_BYTE_OFFSET(alloc.ptr(), alloc.chunk_size() / 2),
            alloc.size() - alloc.chunk_size(), &range));
    meta = vmm_multi_meta(memh);
    ASSERT_TRUE(meta != NULL);

    EXPECT_EQ((CUdeviceptr)alloc.ptr(), range.d_bptr);
    EXPECT_EQ(alloc.size(), range.b_len);
    EXPECT_EQ(UCT_CUDA_IPC_VMM_MULTI_VERSION, range.info.version);
    EXPECT_EQ(UCT_CUDA_IPC_VMM_MULTI_HANDLE_FABRIC, range.info.handle_type);
    EXPECT_EQ(0u, range.info.first_chunk);
    EXPECT_EQ(num_chunks, range.info.num_chunks);
    EXPECT_EQ(meta->alloc_size, range.info.meta_size);
    EXPECT_EQ(num_chunks, meta->num_chunks);
    EXPECT_NE(0ull, meta->buffer_id);

    /* The written slots name every allocation, in order */
    ASSERT_EQ(CUDA_SUCCESS,
              cuMemcpyDtoH(descs.data(), meta->dev_ptr,
                           num_chunks * sizeof(descs[0])));
    for (unsigned i = 0; i < num_chunks; ++i) {
        chunk = (CUdeviceptr)alloc.ptr() + (i * alloc.chunk_size());
        EXPECT_EQ(chunk, descs[i].d_bptr);
        EXPECT_EQ(alloc.chunk_size(), descs[i].b_len);
        ASSERT_EQ(CUDA_SUCCESS,
                  cuPointerGetAttribute(&buffer_id,
                                        CU_POINTER_ATTRIBUTE_BUFFER_ID, chunk));
        EXPECT_EQ(buffer_id, descs[i].buffer_id);
    }

    /* A sub-range is described by its own slots of the same metadata */
    ASSERT_NO_FATAL_FAILURE(vmm_multi_publish(
            memh, UCS_PTR_BYTE_OFFSET(alloc.ptr(), alloc.chunk_size()),
            2 * alloc.chunk_size(), &sub_range));
    EXPECT_EQ(meta, vmm_multi_meta(memh));
    EXPECT_EQ(1u, sub_range.info.first_chunk);
    EXPECT_EQ(2u, sub_range.info.num_chunks);
    EXPECT_EQ((CUdeviceptr)alloc.ptr() + alloc.chunk_size(), sub_range.d_bptr);
    EXPECT_EQ(2 * alloc.chunk_size(), sub_range.b_len);

    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_export_on_demand)
{
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 8;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    const uct_cuda_ipc_vmm_multi_meta_t *meta;
    uct_cuda_ipc_vmm_multi_range_t range;
    uct_mem_h memh;

    auto chunk = [&](unsigned i) {
        return UCS_PTR_BYTE_OFFSET(alloc.ptr(), i * alloc.chunk_size());
    };
    auto exported = [&]() {
        std::vector<uint8_t> bits;

        for (unsigned i = 0; i < meta->num_chunks; ++i) {
            bits.push_back(ucs_dynamic_bitmap_get(&meta->exported, i));
        }
        return bits;
    };

    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));

    /* Only the chunks a pack covers are exported */
    ASSERT_NO_FATAL_FAILURE(vmm_multi_publish(memh, chunk(2),
                                              2 * alloc.chunk_size(), &range));
    meta = vmm_multi_meta(memh);
    ASSERT_TRUE(meta != NULL);
    EXPECT_EQ(num_chunks, meta->num_chunks);
    EXPECT_EQ(std::vector<uint8_t>({0, 0, 1, 1, 0, 0, 0, 0}), exported());

    /* An overlapping pack exports only its new chunks */
    ASSERT_NO_FATAL_FAILURE(vmm_multi_publish(memh, chunk(3),
                                              3 * alloc.chunk_size(), &range));
    EXPECT_EQ(std::vector<uint8_t>({0, 0, 1, 1, 1, 1, 0, 0}), exported());

    /* Growing prefixes keep using the one metadata buffer */
    for (unsigned n = 2; n <= num_chunks; ++n) {
        ASSERT_NO_FATAL_FAILURE(vmm_multi_publish(memh, alloc.ptr(),
                                                  n * alloc.chunk_size(),
                                                  &range));
        EXPECT_EQ(meta, vmm_multi_meta(memh));
        EXPECT_EQ(n, range.info.num_chunks);
    }
    EXPECT_EQ(std::vector<uint8_t>(num_chunks, 1), exported());

    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_fetch_chunks) {
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 4;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    uct_cuda_ipc_vmm_chunk_desc_t *chunks;
    uct_cuda_ipc_rkey_t key;
    uct_mem_h memh;
    CUdevice cu_dev;

    ASSERT_UCS_OK(
            md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL, &memh));
    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));

    /* A key for chunks 1..2 reads only its own slots */
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_key(memh,
                          UCS_PTR_BYTE_OFFSET(alloc.ptr(), alloc.chunk_size()),
                          2 * alloc.chunk_size(), &key));
    ASSERT_UCS_OK(uct_cuda_ipc_vmm_multi_fetch_chunks(&key, cu_dev, &chunks,
                                                      UCS_LOG_LEVEL_ERROR));
    for (unsigned i = 0; i < 2; ++i) {
        EXPECT_EQ((CUdeviceptr)alloc.ptr() + ((i + 1) * alloc.chunk_size()),
                  chunks[i].d_bptr);
        EXPECT_EQ(alloc.chunk_size(), chunks[i].b_len);
    }

    ucs_free(chunks);
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_fetch_rejects_bad_layout) {
#if HAVE_CUDA_FABRIC
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, 2);
    uct_cuda_ipc_vmm_chunk_desc_t *chunks;
    uct_cuda_ipc_rkey_t key, bad_key;
    uct_mem_h memh;
    CUdevice cu_dev;

    ASSERT_UCS_OK(
            md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL, &memh));
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_key(memh, alloc.ptr(), alloc.size(), &key));
    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));

    /* A layout this receiver does not know */
    bad_key = key;
    bad_key.ph.vmm_multi.version++;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              uct_cuda_ipc_vmm_multi_fetch_chunks(&bad_key, cu_dev, &chunks,
                                                  UCS_LOG_LEVEL_DEBUG));

    /* A handle type this receiver does not know */
    bad_key = key;
    bad_key.ph.vmm_multi.handle_type++;
    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              uct_cuda_ipc_vmm_multi_fetch_chunks(&bad_key, cu_dev, &chunks,
                                                  UCS_LOG_LEVEL_DEBUG));

    /* A key which covers no chunks */
    bad_key                         = key;
    bad_key.ph.vmm_multi.num_chunks = 0;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_cuda_ipc_vmm_multi_fetch_chunks(&bad_key, cu_dev, &chunks,
                                                  UCS_LOG_LEVEL_DEBUG));

    /* Slots beyond the metadata buffer */
    bad_key                          = key;
    bad_key.ph.vmm_multi.first_chunk = key.ph.vmm_multi.meta_size /
                                       sizeof(uct_cuda_ipc_vmm_chunk_desc_t);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_cuda_ipc_vmm_multi_fetch_chunks(&bad_key, cu_dev, &chunks,
                                                  UCS_LOG_LEVEL_DEBUG));

    /* A key base which is not the base of its first chunk */
    bad_key         = key;
    bad_key.d_bptr += alloc.chunk_size() / 2;
    bad_key.b_len  -= alloc.chunk_size() / 2;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_cuda_ipc_vmm_multi_fetch_chunks(&bad_key, cu_dev, &chunks,
                                                  UCS_LOG_LEVEL_DEBUG));

    /* Descriptors which do not cover the key range */
    bad_key        = key;
    bad_key.b_len -= alloc.chunk_size();
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              uct_cuda_ipc_vmm_multi_fetch_chunks(&bad_key, cu_dev, &chunks,
                                                  UCS_LOG_LEVEL_DEBUG));

    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_import_round_trip) {
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 4;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    std::vector<uint8_t> sent = vmm_multi_fill(alloc);
    uct_cuda_ipc_rkey_t key;
    CUdeviceptr mapped;
    uct_mem_h memh;
    CUdevice cu_dev;

    ASSERT_UCS_OK(
            md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL, &memh));
    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));

    /* The chunks of a key for chunks 1..2 land at their offsets of one
     * reservation sized for the key */
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_key(memh,
                          UCS_PTR_BYTE_OFFSET(alloc.ptr(), alloc.chunk_size()),
                          2 * alloc.chunk_size(), &key));
    ASSERT_UCS_OK(uct_cuda_ipc_open_memhandle_vmm_multi(&key, cu_dev, &mapped,
                                                        UCS_LOG_LEVEL_ERROR));
    EXPECT_EQ(vmm_multi_expected(sent, alloc, key),
              read_device((void*)mapped, key.b_len));

    EXPECT_EQ(CUDA_SUCCESS, cuMemUnmap(mapped, key.b_len));
    EXPECT_EQ(CUDA_SUCCESS, cuMemAddressFree(mapped, key.b_len));
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_import_without_context) {
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 2;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    std::vector<uint8_t> sent = vmm_multi_fill(alloc);
    ucs_status_t status       = UCS_ERR_LAST;
    CUcontext thread_ctx      = NULL;
    CUdeviceptr mapped        = 0;
    uct_cuda_ipc_rkey_t key;
    uct_mem_h memh;
    CUdevice cu_dev;

    ASSERT_UCS_OK(
            md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL, &memh));
    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_key(memh, alloc.ptr(), alloc.size(), &key));

    /* Like ucp_rkey_ptr() from a thread which never made a context current:
     * the key is opened for cu_dev while no context is current */
    std::thread([&]() {
        cuCtxGetCurrent(&thread_ctx);
        status = uct_cuda_ipc_open_memhandle_vmm_multi(&key, cu_dev, &mapped,
                                                       UCS_LOG_LEVEL_DEBUG);
    }).join();

    ASSERT_TRUE(thread_ctx == NULL) << "the new thread has a current context";
    ASSERT_UCS_OK(status);
    EXPECT_EQ(vmm_multi_expected(sent, alloc, key),
              read_device((void*)mapped, key.b_len));

    EXPECT_EQ(CUDA_SUCCESS, cuMemUnmap(mapped, key.b_len));
    EXPECT_EQ(CUDA_SUCCESS, cuMemAddressFree(mapped, key.b_len));
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_view_cache)
{
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 2;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    uct_cuda_ipc_cache_region_t *region, *new_region;
    uct_cuda_ipc_extended_rkey_t rkey, new_rkey;
    uct_cuda_ipc_rkey_t key, new_key;
    void *mapped, *new_mapped;
    std::vector<uint8_t> sent;
    uct_mem_h memh, new_memh;
    CUdevice cu_dev;

    if (!uct_cuda_ipc_component.enable_remote_cache) {
        UCS_TEST_SKIP_R("CUDA IPC cache is disabled");
    }

    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));
    sent = vmm_multi_fill(alloc);
    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));
    ASSERT_NO_FATAL_FAILURE(vmm_multi_key(memh, alloc.ptr(), alloc.size(),
                                          &key));
    rkey = vmm_multi_remote_key(key);
    ASSERT_NO_FATAL_FAILURE(vmm_multi_map(&rkey, cu_dev, &mapped, &region));

    /* Registering the same memory again creates a new metadata buffer */
    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &new_memh));
    ASSERT_NO_FATAL_FAILURE(vmm_multi_key(new_memh, alloc.ptr(), alloc.size(),
                                          &new_key));
    EXPECT_NE(key.ph.buffer_id, new_key.ph.buffer_id);
    new_rkey = vmm_multi_remote_key(new_key);
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_map(&new_rkey, cu_dev, &new_mapped, &new_region));
    EXPECT_NE(region, new_region);

    /* The replaced view stays mapped until its last reference is released */
    EXPECT_EQ(sent, read_device(mapped, alloc.size()));
    vmm_multi_unmap(&rkey, cu_dev, mapped, region, 1);
    EXPECT_FALSE(vmm_is_mapped(mapped));

    /* Releasing an unused key which holds no mapping frees the cached view */
    uct_cuda_ipc_unmap_unused_view(&new_rkey, cu_dev);
    EXPECT_TRUE(vmm_is_mapped(new_mapped));
    vmm_multi_unmap(&new_rkey, cu_dev, new_mapped, new_region, 1);
    EXPECT_TRUE(vmm_is_mapped(new_mapped));
    uct_cuda_ipc_unmap_unused_view(&new_rkey, cu_dev);
    EXPECT_FALSE(vmm_is_mapped(new_mapped));

    mem_dereg(new_memh);
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_view_overlap)
{
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 4;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    uct_cuda_ipc_cache_region_t *region_a, *region_b, *region_c, *region;
    void *mapped_a, *mapped_b, *mapped_c, *mapped;
    uct_cuda_ipc_extended_rkey_t rkey_a, rkey_b, rkey_c;
    uct_cuda_ipc_rkey_t key_a, key_b, key_c;
    std::vector<uint8_t> sent;
    uct_mem_h memh;
    CUdevice cu_dev;

    if (!uct_cuda_ipc_component.enable_remote_cache) {
        UCS_TEST_SKIP_R("CUDA IPC cache is disabled");
    }

    auto chunk = [&](unsigned i) {
        return UCS_PTR_BYTE_OFFSET(alloc.ptr(), i * alloc.chunk_size());
    };

    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));
    sent = vmm_multi_fill(alloc);
    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));

    /* Packs of one registration for A=[C0,C2], B=[C1,C3] and C=[C0,C3] */
    ASSERT_NO_FATAL_FAILURE(vmm_multi_key(memh, chunk(0),
                                          3 * alloc.chunk_size(), &key_a));
    ASSERT_NO_FATAL_FAILURE(vmm_multi_key(memh, chunk(1),
                                          3 * alloc.chunk_size(), &key_b));
    ASSERT_NO_FATAL_FAILURE(vmm_multi_key(memh, chunk(0), alloc.size(),
                                          &key_c));
    rkey_a = vmm_multi_remote_key(key_a);
    rkey_b = vmm_multi_remote_key(key_b);
    rkey_c = vmm_multi_remote_key(key_c);

    /* Overlapping keys of one metadata buffer get their own views, which
     * coexist */
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_map(&rkey_a, cu_dev, &mapped_a, &region_a));
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_map(&rkey_b, cu_dev, &mapped_b, &region_b));
    EXPECT_NE(region_a, region_b);

    ASSERT_NO_FATAL_FAILURE(vmm_multi_map(&rkey_a, cu_dev, &mapped, &region));
    EXPECT_EQ(region_a, region);
    EXPECT_EQ(mapped_a, mapped);
    vmm_multi_unmap(&rkey_a, cu_dev, mapped, region, 1);

    /* A view containing both replaces them, while they stay mapped for the
     * operations holding them */
    ASSERT_NO_FATAL_FAILURE(
            vmm_multi_map(&rkey_c, cu_dev, &mapped_c, &region_c));
    EXPECT_NE(region_a, region_c);
    EXPECT_NE(region_b, region_c);
    EXPECT_EQ(vmm_multi_expected(sent, alloc, key_a),
              read_device(mapped_a, key_a.b_len));
    EXPECT_EQ(vmm_multi_expected(sent, alloc, key_b),
              read_device(mapped_b, key_b.b_len));

    ASSERT_NO_FATAL_FAILURE(vmm_multi_map(&rkey_b, cu_dev, &mapped, &region));
    EXPECT_EQ(region_c, region);
    EXPECT_EQ(UCS_PTR_BYTE_OFFSET(mapped_c, alloc.chunk_size()), mapped);
    vmm_multi_unmap(&rkey_b, cu_dev, mapped, region, 0);

    vmm_multi_unmap(&rkey_a, cu_dev, mapped_a, region_a, 0);
    vmm_multi_unmap(&rkey_b, cu_dev, mapped_b, region_b, 0);
    vmm_multi_unmap(&rkey_c, cu_dev, mapped_c, region_c, 0);
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mpack_vmm_multi)
{
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 4;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    uct_cuda_ipc_extended_rkey_t rkey, sub_rkey;
    uct_mem_h memh;

    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));
    ASSERT_UCS_OK(md()->ops->mkey_pack(md(), memh, alloc.ptr(), alloc.size(),
                                       NULL, &rkey));
    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI,
              rkey.super.ph.handle_type);
    EXPECT_EQ((CUdeviceptr)alloc.ptr(), rkey.super.d_bptr);
    EXPECT_EQ(alloc.size(), rkey.super.b_len);
    EXPECT_EQ(0u, rkey.super.ph.vmm_multi.first_chunk);
    EXPECT_EQ(num_chunks, rkey.super.ph.vmm_multi.num_chunks);
    ASSERT_TRUE(vmm_multi_meta(memh) != NULL);
    EXPECT_EQ(vmm_multi_meta(memh)->buffer_id, rkey.super.ph.buffer_id);

    /* A pack starting in a later chunk covers its own chunks of the same
     * published layout */
    ASSERT_UCS_OK(md()->ops->mkey_pack(
            md(), memh, UCS_PTR_BYTE_OFFSET(alloc.ptr(), alloc.chunk_size()),
            2 * alloc.chunk_size(), NULL, &sub_rkey));
    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI,
              sub_rkey.super.ph.handle_type);
    EXPECT_EQ(rkey.super.ph.buffer_id, sub_rkey.super.ph.buffer_id);
    EXPECT_EQ(1u, sub_rkey.super.ph.vmm_multi.first_chunk);
    EXPECT_EQ(2u, sub_rkey.super.ph.vmm_multi.num_chunks);
    EXPECT_EQ((CUdeviceptr)alloc.ptr() + alloc.chunk_size(),
              sub_rkey.super.d_bptr);
    EXPECT_EQ(2 * alloc.chunk_size(), sub_rkey.super.b_len);

    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mpack_vmm_single_allocation)
{
#if HAVE_CUDA_FABRIC
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, 2);
    uct_cuda_ipc_extended_rkey_t rkey;
    uct_mem_h memh;

    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));
    /* A range inside one allocation still packs a plain VMM key */
    ASSERT_UCS_OK(md()->ops->mkey_pack(md(), memh, alloc.ptr(),
                                       alloc.chunk_size(), NULL, &rkey));
    EXPECT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM, rkey.super.ph.handle_type);
    EXPECT_TRUE(vmm_multi_meta(memh) == NULL);
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, rkey_unpack_vmm_multi_unsupported)
{
#if HAVE_CUDA_FABRIC
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, 2);
    uct_rkey_unpack_params_t unpack_params = {};
    uct_cuda_ipc_extended_rkey_t rkey, bad_rkey;
    uct_rkey_bundle_t rkey_bundle;
    ucs_status_t status;
    uct_mem_h memh;

    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));
    ASSERT_UCS_OK(md()->ops->mkey_pack(md(), memh, alloc.ptr(), alloc.size(),
                                       NULL, &rkey));
    ASSERT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI,
              rkey.super.ph.handle_type);

    /* Keys of a format this receiver does not know are unreachable */
    bad_rkey = rkey;
    bad_rkey.super.ph.vmm_multi.handle_type++;
    EXPECT_EQ(UCS_ERR_UNREACHABLE,
              uct_rkey_unpack_v2(md()->component, &bad_rkey, &unpack_params,
                                 &rkey_bundle));

    bad_rkey = rkey;
    bad_rkey.super.ph.vmm_multi.version++;
    EXPECT_EQ(UCS_ERR_UNREACHABLE,
              uct_rkey_unpack_v2(md()->component, &bad_rkey, &unpack_params,
                                 &rkey_bundle));

    /* Rejecting them leaves no cached state for the supported keys */
    status = uct_rkey_unpack_v2(md()->component, &rkey, &unpack_params,
                                &rkey_bundle);
    if (status != UCS_ERR_UNREACHABLE) {
        ASSERT_UCS_OK(status);
        EXPECT_UCS_OK(uct_rkey_release(md()->component, &rkey_bundle));
    }

    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_vmm_multi)
{
#if HAVE_CUDA_FABRIC
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, 4);

    test_mkey_pack_on_thread(alloc.ptr(), alloc.size());
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_pack_map_round_trip)
{
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 4;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    uct_cuda_ipc_cache_region_t *region;
    uct_cuda_ipc_extended_rkey_t rkey;
    std::vector<uint8_t> sent;
    uct_mem_h memh;
    CUdevice cu_dev;
    void *mapped;

    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));
    sent = vmm_multi_fill(alloc);
    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));
    ASSERT_UCS_OK(md()->ops->mkey_pack(
            md(), memh, UCS_PTR_BYTE_OFFSET(alloc.ptr(), alloc.chunk_size()),
            2 * alloc.chunk_size(), NULL, &rkey));
    ASSERT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI,
              rkey.super.ph.handle_type);

    /* Map the packed key as another process would */
    rkey = vmm_multi_remote_key(rkey.super);
    vmm_multi_map(&rkey, cu_dev, &mapped, &region);
    EXPECT_EQ(vmm_multi_expected(sent, alloc, rkey.super),
              read_device(mapped, rkey.super.b_len));

    vmm_multi_unmap(&rkey, cu_dev, mapped, region, 0);
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, vmm_multi_rkey_release_frees_view)
{
#if HAVE_CUDA_FABRIC
    constexpr unsigned num_chunks = 3;
    cuda_fabric_mem_buffer alloc(1, UCS_MEMORY_TYPE_CUDA, num_chunks);
    uct_rkey_unpack_params_t unpack_params = {};
    uct_cuda_ipc_cache_region_t *region, *single_region;
    uct_cuda_ipc_extended_rkey_t rkey, single_rkey;
    uct_cuda_ipc_unpacked_rkey_t *unpacked;
    void *mapped, *single_mapped;
    uct_rkey_bundle_t rkey_bundle;
    CUdeviceptr base;
    uct_mem_h memh;
    CUdevice cu_dev;
    size_t size;

    if (!uct_cuda_ipc_component.enable_remote_cache) {
        UCS_TEST_SKIP_R("CUDA IPC cache is disabled");
    }

    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_dev));
    ASSERT_UCS_OK(md()->ops->mem_reg(md(), alloc.ptr(), alloc.size(), NULL,
                                     &memh));
    ASSERT_UCS_OK(md()->ops->mkey_pack(md(), memh, alloc.ptr(), alloc.size(),
                                       NULL, &rkey));
    ASSERT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI,
              rkey.super.ph.handle_type);

    /* A single-allocation key at the same base, as packed for a range inside
     * the first chunk */
    ASSERT_UCS_OK(md()->ops->mkey_pack(md(), memh, alloc.ptr(),
                                       alloc.chunk_size(), NULL, &single_rkey));
    ASSERT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM, single_rkey.super.ph.handle_type);
    ASSERT_EQ(rkey.super.d_bptr, single_rkey.super.d_bptr);

    rkey.super.pid           = getpid() + 100000;
    unpack_params.field_mask = UCT_RKEY_UNPACK_FIELD_FLAGS;
    unpack_params.flags      = UCT_RKEY_UNPACK_FLAG_FREE_UNUSED_MAPPING;
    for (int i = 0; i < 2; ++i) {
        ucs_status_t status = uct_rkey_unpack_v2(md()->component, &rkey,
                                                 &unpack_params, &rkey_bundle);
        if (status == UCS_ERR_UNREACHABLE) {
            mem_dereg(memh);
            UCS_TEST_SKIP_R("multi-chunk key is unreachable");
        }

        ASSERT_UCS_OK(status);
        if (i == 0) {
            EXPECT_UCS_OK(uct_rkey_release(md()->component, &rkey_bundle));
        }
    }

    /* An operation maps the key and releases its view to the cache */
    unpacked = (uct_cuda_ipc_unpacked_rkey_t*)rkey_bundle.rkey;
    vmm_multi_map(&unpacked->super, cu_dev, &mapped, &region);
    vmm_multi_unmap(&unpacked->super, cu_dev, mapped, region, 1);

    /* So does an operation on the single-allocation key of the same peer */
    single_rkey.super.pid = unpacked->super.super.pid;
    single_rkey.pid_ns    = unpacked->super.pid_ns;
    ASSERT_UCS_OK(uct_cuda_ipc_map_memhandle(&single_rkey, cu_dev,
                                             &single_mapped, &single_region,
                                             UCS_LOG_LEVEL_ERROR));
    EXPECT_TRUE(single_region == NULL);
    vmm_multi_unmap(&single_rkey, cu_dev, single_mapped, single_region, 1);

    /* Releasing the key unmaps its unused view, and only its view */
    EXPECT_UCS_OK(uct_rkey_release(md()->component, &rkey_bundle));
    EXPECT_NE(CUDA_SUCCESS,
              cuMemGetAddressRange(&base, &size, (CUdeviceptr)mapped));
    EXPECT_EQ(CUDA_SUCCESS,
              cuMemGetAddressRange(&base, &size, (CUdeviceptr)single_mapped));

    /* Drop the cached single-allocation mapping */
    ASSERT_UCS_OK(uct_cuda_ipc_map_memhandle(&single_rkey, cu_dev,
                                             &single_mapped, &single_region,
                                             UCS_LOG_LEVEL_ERROR));
    vmm_multi_unmap(&single_rkey, cu_dev, single_mapped, single_region, 0);
    mem_dereg(memh);
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}

UCS_TEST_P(test_cuda_ipc_md, nvml_peer_accessibility)
{
    int num_devices, can_access, device_index;
    ucs_ternary_auto_value_t accessible;
    CUdevice local_device, remote_device;
    CUuuid remote_uuid;
    bool checked = false;

    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&num_devices));
    if (num_devices < 2) {
        UCS_TEST_SKIP_R("test requires at least two CUDA devices");
    }

    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&local_device, 0));
    for (device_index = 1; device_index < num_devices; ++device_index) {
        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&remote_device, device_index));
        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetUuid(&remote_uuid, remote_device));
        ASSERT_EQ(CUDA_SUCCESS,
                  cuDeviceCanAccessPeer(&can_access, local_device,
                                        remote_device));

        accessible = uct_cuda_ipc_nvml_peer_accessible(local_device,
                                                        &remote_uuid);
        if (accessible != UCS_TRY) {
            EXPECT_EQ(can_access ? UCS_YES : UCS_NO, accessible);
            checked = true;
        }
    }

    if (!checked) {
        UCS_TEST_SKIP_R("NVML peer accessibility is unavailable");
    }
}

UCS_TEST_P(test_cuda_ipc_md, nvml_peer_unresolved)
{
    CUdevice cu_dev;
    CUuuid unknown_uuid;

    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&cu_dev, 0));

    /* NVML cannot resolve this UUID, so the handle-mapping probe must decide
     * the peer reachability */
    memset(unknown_uuid.bytes, 0xab, sizeof(unknown_uuid.bytes));
    EXPECT_EQ(UCS_TRY,
              uct_cuda_ipc_nvml_peer_accessible(cu_dev, &unknown_uuid));
}

UCS_TEST_P(test_cuda_ipc_md, posix_fd_same_node_ipc)
{
#if HAVE_DECL_SYS_PIDFD_GETFD
    cuda_posix_fd_mem_buffer buf(4096, UCS_MEMORY_TYPE_CUDA);
    size_t size = buf.size();
    uct_mem_h memh;
    uct_cuda_ipc_rkey_t rkey = {};

    ASSERT_EQ(CUDA_SUCCESS, cuMemsetD8((CUdeviceptr)buf.ptr(), 0xAB, size));

    ASSERT_UCS_OK(pack_posix_fd_key(buf, size, &memh, &rkey));
    ASSERT_EQ(UCT_CUDA_IPC_KEY_HANDLE_TYPE_POSIX_FD, rkey.ph.handle_type);
    ASSERT_EQ(ucs_get_system_id(), rkey.ph.handle.posix_fd.system_id);

    /* Tamper UUID to bypass same-process shortcut and exercise the full
     * POSIX FD import path (pidfd_open/pidfd_getfd + cuMemImport) */
    tamper_rkey_uuid(&rkey, 0x1234LL, 0x5678LL);

    uct_component_t *component = md()->component;

    /* Unpack on a separate thread with its own CUDA context */
    std::exception_ptr thread_exception;
    std::thread([&]() {
        try {
            CUdevice dev;
            CUcontext ctx;
            ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, 0));
            ASSERT_EQ(CUDA_SUCCESS,
                      uct_test_cuda_ctx_create_compat(&ctx, 0, dev));

            uct_rkey_unpack_params_t unpack_params = {};
            uct_rkey_bundle_t rkey_bundle = {};
            ASSERT_UCS_OK(uct_rkey_unpack_v2(component, &rkey, &unpack_params,
                                             &rkey_bundle));

            uct_cuda_ipc_unpacked_rkey_t *unpacked =
                    (uct_cuda_ipc_unpacked_rkey_t*)rkey_bundle.rkey;
            uct_cuda_ipc_cache_region_t *cache_region;
            void *mapped_addr;
            ucs_status_t map_status = uct_cuda_ipc_map_memhandle(
                    &unpacked->super, dev, &mapped_addr, &cache_region,
                    UCS_LOG_LEVEL_ERROR);
            ASSERT_UCS_OK(map_status);

            std::vector<uint8_t> host_buf(size);
            ASSERT_EQ(CUDA_SUCCESS,
                      cuMemcpyDtoH(host_buf.data(), (CUdeviceptr)mapped_addr,
                                   size));
            for (size_t i = 0; i < size; i++) {
                ASSERT_EQ(0xAB, host_buf[i]) << "Data mismatch at byte " << i;
            }

            uct_cuda_ipc_mapping_t mapping;
            uct_cuda_ipc_mapping_init(&mapping, &unpacked->super,
                                      mapped_addr, cache_region);
            uct_cuda_ipc_unmap_memhandle(&mapping, dev, 0);
            EXPECT_UCS_OK(uct_rkey_release(component, &rkey_bundle));
            EXPECT_EQ(cuCtxDestroy(ctx), CUDA_SUCCESS);
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
#else
    UCS_TEST_SKIP_R("built without pidfd support");
#endif
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_ipc_md, cuda_ipc);


class test_cuda_ipc_cache_lru : public ucs::test {
protected:
    static const size_t REGION_SIZE = UCS_MBYTE * 2;
    static const uintptr_t BASE_ADDR = 0x7f0000000000UL;

    virtual void init() {
        ucs::test::init();
        m_cache = NULL;
        /* Reset global limits to unlimited before each test */
        uct_cuda_ipc_cache_set_global_limits(ULONG_MAX, SIZE_MAX);
    }

    virtual void cleanup() {
        if (m_cache != NULL) {
            drain_cache();
            uct_cuda_ipc_destroy_cache(m_cache, 1);
        }
        uct_cuda_ipc_cache_set_global_limits(ULONG_MAX, SIZE_MAX);
        ucs::test::cleanup();
    }

    void create_cache(unsigned long max_regions, size_t max_size) {
        uct_cuda_ipc_cache_set_global_limits(max_regions, max_size);
        m_max_regions = max_regions;
        m_max_size    = max_size;
        ASSERT_EQ(UCS_OK, uct_cuda_ipc_create_cache(&m_cache, "test_lru"));
    }

    uct_cuda_ipc_cache_region_t *insert_region(size_t index) {
        uct_cuda_ipc_cache_region_t *region;
        uintptr_t addr = BASE_ADDR + (index * REGION_SIZE * 2);
        int ret;

        ret = ucs_posix_memalign((void **)&region,
                                 ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                                 sizeof(uct_cuda_ipc_cache_region_t),
                                 "test_cuda_ipc_cache_region");
        EXPECT_EQ(0, ret);

        region->super.start = ucs_align_down_pow2(addr, UCS_PGT_ADDR_ALIGN);
        region->super.end   = ucs_align_up_pow2(addr + REGION_SIZE,
                                                UCS_PGT_ADDR_ALIGN);
        memset(&region->key, 0, sizeof(region->key));
        region->key.b_len      = REGION_SIZE;
        region->key.d_bptr     = addr;
        region->key.ph.buffer_id = index;
        region->mapped_addr    = (void *)addr;
        region->refcount       = 1;
        region->cu_dev         = 0;

        ucs_status_t status = ucs_pgtable_insert(&m_cache->pgtable,
                                                  &region->super);
        EXPECT_EQ(UCS_OK, status);

        m_cache->num_regions++;
        m_cache->total_size += REGION_SIZE;
        ucs_list_add_tail(&m_cache->lru_list, &region->lru_list);

        return region;
    }

    void release_region(uct_cuda_ipc_cache_region_t *region) {
        ASSERT_GE(region->refcount, 1UL);
        region->refcount--;
    }

    void reacquire_region(uct_cuda_ipc_cache_region_t *region) {
        /* Move to LRU tail (most recently used) */
        ucs_list_del(&region->lru_list);
        ucs_list_add_tail(&m_cache->lru_list, &region->lru_list);
        region->refcount++;
    }

    void destroy_region(uct_cuda_ipc_cache_region_t *region)
    {
        ucs_status_t status = ucs_pgtable_remove(&m_cache->pgtable,
                                                 &region->super);
        ASSERT_EQ(UCS_OK, status);

        ucs_list_del(&region->lru_list);

        ASSERT_GT(m_cache->num_regions, 0UL);
        m_cache->num_regions--;
        m_cache->total_size -= region->key.b_len;

        ucs_free(region);
    }

    void evict_lru() {
        uct_cuda_ipc_cache_region_t *region, *tmp;

        ucs_list_for_each_safe(region, tmp, &m_cache->lru_list, lru_list) {
            if ((m_cache->num_regions <= m_max_regions) &&
                (m_cache->total_size <= m_max_size)) {
                break;
            }

            if (region->refcount > 0) {
                continue;
            }

            ASSERT_EQ(UCS_OK, ucs_pgtable_remove(&m_cache->pgtable,
                                                  &region->super));
            ucs_list_del(&region->lru_list);
            m_cache->num_regions--;
            m_cache->total_size -= region->key.b_len;

            ucs_free(region);
        }
    }

    static void collect_region_cb(const ucs_pgtable_t *pgtable,
                                  ucs_pgt_region_t *pgt_region, void *arg) {
        ucs_list_link_t *list = (ucs_list_link_t *)arg;
        uct_cuda_ipc_cache_region_t *region =
                ucs_derived_of(pgt_region, uct_cuda_ipc_cache_region_t);
        ucs_list_add_tail(list, &region->list);
    }

    void drain_cache() {
        ucs_list_link_t region_list;
        uct_cuda_ipc_cache_region_t *region, *tmp;

        ucs_list_head_init(&region_list);
        ucs_pgtable_purge(&m_cache->pgtable, collect_region_cb, &region_list);
        ucs_list_for_each_safe(region, tmp, &region_list, list) {
            ucs_free(region);
        }

        ucs_list_head_init(&m_cache->lru_list);
        m_cache->num_regions = 0;
        m_cache->total_size  = 0;
    }

    bool pgtable_has(size_t index) {
        uintptr_t addr = BASE_ADDR + (index * REGION_SIZE * 2);
        return ucs_pgtable_lookup(&m_cache->pgtable, addr) != NULL;
    }

    uct_cuda_ipc_cache_t *m_cache;
    unsigned long         m_max_regions;
    size_t                m_max_size;
};

const size_t    test_cuda_ipc_cache_lru::REGION_SIZE;
const uintptr_t test_cuda_ipc_cache_lru::BASE_ADDR;

UCS_TEST_F(test_cuda_ipc_cache_lru, evict_by_count) {
    const unsigned long max_regions = 128;
    const size_t num_insert         = 192;

    create_cache(max_regions, SIZE_MAX);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    for (size_t i = 0; i < num_insert; i++) {
        release_region(regions[i]);
    }

    EXPECT_EQ(num_insert, m_cache->num_regions);

    evict_lru();

    EXPECT_EQ(max_regions, m_cache->num_regions);
    EXPECT_EQ(max_regions * REGION_SIZE, m_cache->total_size);

    for (size_t i = 0; i < num_insert - max_regions; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert - max_regions; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, evict_by_size) {
    const size_t max_size   = REGION_SIZE * 64;
    const size_t num_insert = 100;

    create_cache(ULONG_MAX, max_size);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    for (size_t i = 0; i < num_insert; i++) {
        release_region(regions[i]);
    }

    EXPECT_EQ(num_insert, m_cache->num_regions);
    EXPECT_EQ(num_insert * REGION_SIZE, m_cache->total_size);

    evict_lru();

    size_t expected_regions = max_size / REGION_SIZE;
    EXPECT_EQ(expected_regions, m_cache->num_regions);
    EXPECT_EQ(expected_regions * REGION_SIZE, m_cache->total_size);

    for (size_t i = 0; i < num_insert - expected_regions; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert - expected_regions; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, no_evict_in_use) {
    const unsigned long max_regions = 64;
    const size_t num_insert         = 128;

    create_cache(max_regions, SIZE_MAX);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    /* Release only the first half -- the second half stays in-use */
    for (size_t i = 0; i < num_insert / 2; i++) {
        release_region(regions[i]);
    }

    evict_lru();

    /* Only released regions can be evicted; in-use ones remain */
    EXPECT_EQ(max_regions, m_cache->num_regions);

    for (size_t i = 0; i < num_insert / 2; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert / 2; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, lru_order) {
    const unsigned long max_regions = 128;
    const size_t num_insert         = 256;

    create_cache(max_regions, SIZE_MAX);

    std::vector<uct_cuda_ipc_cache_region_t *> regions(num_insert);
    for (size_t i = 0; i < num_insert; i++) {
        regions[i] = insert_region(i);
    }

    for (size_t i = 0; i < num_insert; i++) {
        release_region(regions[i]);
    }

    /*
     * Reacquire then release the first 64 regions, moving them to
     * the tail of the LRU (most recently used). The eviction should
     * then remove regions [64..191] and keep [0..63] + [192..255].
     */
    for (size_t i = 0; i < 64; i++) {
        reacquire_region(regions[i]);
        release_region(regions[i]);
    }

    evict_lru();

    EXPECT_EQ(max_regions, m_cache->num_regions);

    for (size_t i = 0; i < 64; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
    for (size_t i = 64; i < num_insert - 64; i++) {
        EXPECT_FALSE(pgtable_has(i));
    }
    for (size_t i = num_insert - 64; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, unlimited) {
    const size_t num_insert = 512;

    create_cache(ULONG_MAX, SIZE_MAX);

    for (size_t i = 0; i < num_insert; i++) {
        uct_cuda_ipc_cache_region_t *r = insert_region(i);
        release_region(r);
    }

    evict_lru();

    EXPECT_EQ(num_insert, m_cache->num_regions);
    EXPECT_EQ(num_insert * REGION_SIZE, m_cache->total_size);
    for (size_t i = 0; i < num_insert; i++) {
        EXPECT_TRUE(pgtable_has(i));
    }
}

UCS_TEST_F(test_cuda_ipc_cache_lru, stale_destroy_while_in_use) {
    /* Check that evict_lru does not pull in-use regions off
     * the LRU. This could cause a failure if region is destroyed
     * while not in LRU. */
    create_cache(0 /* force eviction on every insert */, SIZE_MAX);

    uct_cuda_ipc_cache_region_t *r1 = insert_region(0);
    ASSERT_EQ(1UL, r1->refcount);

    evict_lru();

    /* Core assertion: in-use region must stay on LRU after eviction */
    EXPECT_EQ(1UL, m_cache->num_regions);
    EXPECT_TRUE(pgtable_has(0));

    /* Simulate the stale-buffer_id destroy path */
    destroy_region(r1);

    EXPECT_EQ(0UL, m_cache->num_regions);
    EXPECT_EQ(0UL, m_cache->total_size);
    EXPECT_TRUE(ucs_list_is_empty(&m_cache->lru_list));
}

UCS_TEST_F(test_cuda_ipc_cache_lru, destroy_with_views) {
#if HAVE_CUDA_FABRIC
    ucs_memtrack_entry_t before, after;

    ucs_memtrack_cleanup();
    push_config();
    modify_config("MEMTRACK_DEST", "/dev/null");
    ucs_memtrack_init();
    ucs_memtrack_total(&before);

    create_cache(ULONG_MAX, SIZE_MAX);
    insert_region(0);

    /* Views in use: one in the cache, and one replaced but still mapped */
    for (int in_map = 0; in_map <= 1; ++in_map) {
        uct_cuda_ipc_cache_view_t *view = (uct_cuda_ipc_cache_view_t*)
                ucs_calloc(1, sizeof(*view), "test_cuda_ipc_cache_view");
        ASSERT_TRUE(view != NULL);

        view->super.key.ph.handle_type = UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM_MULTI;
        view->super.key.d_bptr         = BASE_ADDR;
        view->super.key.b_len          = REGION_SIZE;
        view->super.refcount           = 1;
        view->in_map                   = in_map;
        if (in_map) {
            ucs_interval_map_insert(&m_cache->views, &view->node, BASE_ADDR,
                                    BASE_ADDR + REGION_SIZE);
        }

        m_cache->num_regions++;
        m_cache->total_size += REGION_SIZE;
        ucs_list_add_tail(&m_cache->lru_list, &view->super.lru_list);
    }

    /* The page table purge must leave the views to the LRU walk */
    uct_cuda_ipc_destroy_cache(m_cache, 0);
    m_cache = NULL;

    ucs_memtrack_total(&after);
    EXPECT_EQ(before.count, after.count);
    ucs_memtrack_cleanup();
    pop_config();
    ucs_memtrack_init();
#else
    UCS_TEST_SKIP_R("built without fabric support");
#endif
}
