/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_custom_buffer.h"
#include <common/cuda_context.h>
#include <uct/test_md.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
#include <ucs/sys/ptr_arith.h>
}

#include <cuda.h>

#include <thread>

class test_cuda_ipc_md : public test_md {
protected:
    static uct_cuda_ipc_rkey_t
    unpack_common(uct_md_h md, int64_t uuid, CUdeviceptr ptr, size_t size)
    {
        uct_mem_h memh;
        uct_cuda_ipc_rkey_t rkey;
        EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, size, NULL, &memh));
        EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, size, NULL,
                                         &rkey));

        int64_t *uuid64 = (int64_t *)rkey.uuid.bytes;
        uuid64[0]       = uuid;
        uuid64[1]       = uuid;

        /* cuIpcOpenMemHandle used by cuda_ipc_cache does not allow to open
         * handle that was created by the same process */
        EXPECT_EQ(UCS_ERR_UNREACHABLE,
                  md->component->rkey_unpack(md->component, &rkey, NULL, NULL));

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
    static uct_cuda_ipc_rkey_t unpack_masync(uct_md_h md, int64_t uuid)
    {
        size_t size = 4 * UCS_MBYTE;
        CUdeviceptr ptr;
        CUmemoryPool mpool;
        CUstream cu_stream;

        cuda_mem_pool::alloc_mempool(&ptr, &mpool, &cu_stream, size);
        uct_cuda_ipc_rkey_t rkey = unpack_common(md, uuid, ptr, size);
        cuda_mem_pool::free_mempool(&ptr, &mpool, &cu_stream);
        return rkey;
    }
#endif

    template<class T> void mkey_pack() const;
};

template<class T> void test_cuda_ipc_md::mkey_pack() const
{
    const size_t size = 16;
    T buffer(size, UCS_MEMORY_TYPE_CUDA);

    uct_md_mem_reg_params_t reg_params = {};
    uct_mem_h memh;
    ASSERT_UCS_OK(uct_md_mem_reg_v2(md(), buffer.ptr(), size, &reg_params,
                                    &memh));

    std::exception_ptr thread_exception;
    std::thread([&]() {
        try {
            uct_md_mkey_pack_params_t params = {};
            std::vector<uint8_t> rkey(md_attr().rkey_packed_size);
            ASSERT_UCS_OK(uct_md_mkey_pack_v2(md(), memh, buffer.ptr(),
                                              size, &params, rkey.data()));
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

UCS_TEST_P(test_cuda_ipc_md, missing_device_context)
{
    cuda_context cuda_ctx;
    ucs_status_t status;
    uct_cuda_ipc_rkey_t rkey;
    ucs::handle<uct_md_h> md;
    int dev_num;

    UCS_TEST_CREATE_HANDLE(uct_md_h, md, uct_md_close, uct_md_open,
                           GetParam().component, GetParam().md_name.c_str(),
                           m_md_config);

    // CUDA IPC cache is functional
    rkey          = unpack(md, 1);
    dev_num       = rkey.dev_num;
    rkey.dev_num  = ~rkey.dev_num;
    rkey          = unpack(md, 1);
    EXPECT_EQ(dev_num, rkey.dev_num);

    // Unpack without a CUDA device context
    std::thread t([&md, &rkey, &status]() {
        rkey.dev_num = ~rkey.dev_num;
        status = md->component->rkey_unpack(md->component, &rkey, NULL, NULL);
    });
    t.join();

    EXPECT_EQ(UCS_ERR_UNREACHABLE, status);
    EXPECT_NE(dev_num, rkey.dev_num); // rkey was not updated
}

UCS_MT_TEST_P(test_cuda_ipc_md, multiple_mds, 8)
{
    cuda_context cuda_ctx;
    ucs::handle<uct_md_h> md;
    UCS_TEST_CREATE_HANDLE(uct_md_h, md, uct_md_close, uct_md_open,
                           GetParam().component, GetParam().md_name.c_str(),
                           m_md_config);

    {
        /* Create and destroy temporary MD */
        ucs::handle<uct_md_h> tmp_md;
        UCS_TEST_CREATE_HANDLE(uct_md_h, tmp_md, uct_md_close, uct_md_open,
                               GetParam().component, GetParam().md_name.c_str(),
                               m_md_config);
    }

    for (int64_t i = 0; i < 64; ++i) {
        /* We get unique dev_num on new UUID */
        uct_cuda_ipc_rkey_t rkey = unpack(md, i + 1);
        EXPECT_EQ(i, rkey.dev_num);
        /* Subsequent call with the same UUID returns value from cache */
        rkey = unpack(md, i + 1);
        EXPECT_EQ(i, rkey.dev_num);
    }
}

#if HAVE_CUDA_FABRIC
UCS_MT_TEST_P(test_cuda_ipc_md, multiple_mds_mempool, 8)
{
    cuda_context cuda_ctx;
    ucs::handle<uct_md_h> md;
    UCS_TEST_CREATE_HANDLE(uct_md_h, md, uct_md_close, uct_md_open,
                           GetParam().component, GetParam().md_name.c_str(),
                           m_md_config);

    CUdeviceptr ptr;
    CUmemoryPool mpool, q_mpool;
    CUstream cu_stream;
    CUmemFabricHandle fabric_handle;
    CUresult cu_err;

    cuda_mem_pool::alloc_mempool(&ptr, &mpool, &cu_stream, 64);
    EXPECT_EQ(CUDA_SUCCESS, (cuPointerGetAttribute((void*)&q_mpool,
                    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, ptr)));

    cu_err = cuMemPoolExportToShareableHandle((void*)&fabric_handle, q_mpool,
                                              CU_MEM_HANDLE_TYPE_FABRIC, 0);
    cuda_mem_pool::free_mempool(&ptr, &mpool, &cu_stream);

    if (cu_err == CUDA_SUCCESS) {
        for (int64_t i = 0; i < 64; ++i) {
            /* We get unique dev_num on new UUID */
            uct_cuda_ipc_rkey_t rkey = unpack_masync(md, i + 1);
            EXPECT_EQ(i, rkey.dev_num);
            /* Subsequent call with the same UUID returns value from cache */
            rkey = unpack_masync(md, i + 1);
            EXPECT_EQ(i, rkey.dev_num);
        }
    }
}
#endif

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_legacy)
{
    mkey_pack<mem_buffer>();
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_error)
{
    mkey_pack<cuda_vmm_mem_buffer>();
}

#if HAVE_CUDA_FABRIC
UCS_TEST_P(test_cuda_ipc_md, mkey_pack_mempool)
{
    mkey_pack<cuda_mem_pool>();
}

UCS_TEST_P(test_cuda_ipc_md, mkey_pack_fabric)
{
    mkey_pack<cuda_fabric_mem_buffer>();
}
#endif

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_ipc_md, cuda_ipc);
