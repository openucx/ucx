/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/cuda_context.h>
#include <uct/test_md.h>
#include <cuda.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
}

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
};

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

    alloc_mempool(&ptr, &mpool, &cu_stream, 64);
    EXPECT_EQ(CUDA_SUCCESS, (cuPointerGetAttribute((void*)&q_mpool,
                    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, ptr)));

    cu_err = cuMemPoolExportToShareableHandle((void*)&fabric_handle, q_mpool,
                                              CU_MEM_HANDLE_TYPE_FABRIC, 0);
    free_mempool(&ptr, &mpool, &cu_stream);

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

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_ipc_md, cuda_ipc);
