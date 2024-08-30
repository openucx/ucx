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
    static uct_cuda_ipc_rkey_t unpack(uct_md_h md, int64_t uuid)
    {
        CUdeviceptr ptr;
        EXPECT_EQ(CUDA_SUCCESS, cuMemAlloc(&ptr, 64));
        uct_mem_h memh;
        EXPECT_UCS_OK(md->ops->mem_reg(md, (void *)ptr, 64, NULL, &memh));
        uct_cuda_ipc_rkey_t rkey;
        EXPECT_UCS_OK(md->ops->mkey_pack(md, memh, (void *)ptr, 64, NULL,
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

        EXPECT_EQ(CUDA_SUCCESS, cuMemFree(ptr));
        return rkey;
    }
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

_UCT_MD_INSTANTIATE_TEST_CASE(test_cuda_ipc_md, cuda_ipc);
