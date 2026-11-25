/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TEST_UCP_MEMHEAP_H
#define TEST_UCP_MEMHEAP_H

#include "ucp_test.h"

#include <common/mem_buffer.h>


class test_ucp_memheap : public ucp_test {
public:
    /*
     * Function type for memheap send operation
     *
     * @param [in]  size            Size of data to send
     * @param [in]  expected_data   Buffer to fill with expected data at 'target_ptr'
     * @param [in]  memh            Memory handle of local buffer
     * @param [in]  target_ptr      VA to perform the RMA operation to
     * @param [in]  rkey            RMA remote key
     * @param [in]  arg             User-defined argument
     *
     * @note The expected data buffer memory type is 'send_mem_type' as passed
     *       to @ref test_xfer
     * @note The expected_data buffer does not have to be filled during the
     *       function itself, however it must be there after endpoint/worker flush.
     */
    typedef void
    (test_ucp_memheap::* send_func_t)(size_t size, void *expected_data,
                                      ucp_mem_h memh, void *target_ptr,
                                      ucp_rkey_h rkey, void *arg);

protected:
    struct sgl_ctx {
        std::vector<mapped_buffer>           src;
        std::vector<mapped_buffer>           dst;
        std::vector<ucs::handle<ucp_rkey_h>> rkey_handles;
        std::vector<void*>                   buffers;
        std::vector<uint64_t>                remote_addrs;
        std::vector<size_t>                  lengths;
        std::vector<size_t>                  remote_lengths;
        std::vector<ucp_mem_h>               memhs;
        std::vector<ucp_rkey_h>              rkeys;
        uint64_t                             null_local_arrays  = 0;
        uint64_t                             null_remote_arrays = 0;
    };

    void init_sgl_ctx(sgl_ctx &ctx, const std::vector<size_t> &elem_sizes,
                       ucs_memory_type_t src_type, ucs_memory_type_t dst_type,
                       int ep_index = 0);

    void init_sgl_ctx(sgl_ctx &ctx, const std::vector<size_t> &elem_sizes,
                       const std::vector<ucs_memory_type_t> &src_types,
                       const std::vector<ucs_memory_type_t> &dst_types,
                       int ep_index = 0);

    static constexpr uint64_t LOCAL_MASK_DEFAULT =
            UCP_DT_LOCAL_SGL_FIELD_BUFFERS | UCP_DT_LOCAL_SGL_FIELD_LENGTHS;

    static constexpr uint64_t REMOTE_MASK_DEFAULT =
            UCP_DT_REMOTE_SGL_FIELD_REMOTE_ADDRS |
            UCP_DT_REMOTE_SGL_FIELD_LENGTHS |
            UCP_DT_REMOTE_SGL_FIELD_RKEYS;

    template<typename T>
    static const T *sgl_array(const std::vector<T> &array, uint64_t null_arrays,
                              uint64_t field) {
        return (null_arrays & field) ? nullptr : array.data();
    }

    static ucp_dt_local_sgl_t
    make_local_sgl(sgl_ctx &ctx, uint64_t field_mask) {
        ucp_dt_local_sgl_t sgl = {};
        sgl.field_mask         = field_mask;
        sgl.buffers            = sgl_array(ctx.buffers, ctx.null_local_arrays,
                                           UCP_DT_LOCAL_SGL_FIELD_BUFFERS);
        sgl.lengths            = sgl_array(ctx.lengths, ctx.null_local_arrays,
                                           UCP_DT_LOCAL_SGL_FIELD_LENGTHS);
        sgl.memhs              = sgl_array(ctx.memhs, ctx.null_local_arrays,
                                            UCP_DT_LOCAL_SGL_FIELD_MEMHS);
        return sgl;
    }

    static ucp_dt_remote_sgl_t
    make_remote_sgl(sgl_ctx &ctx, uint64_t field_mask) {
        ucp_dt_remote_sgl_t sgl = {};
        sgl.field_mask          = field_mask;
        sgl.remote_addrs        = sgl_array(
                ctx.remote_addrs, ctx.null_remote_arrays,
                UCP_DT_REMOTE_SGL_FIELD_REMOTE_ADDRS);
        sgl.lengths             = sgl_array(ctx.remote_lengths,
                                            ctx.null_remote_arrays,
                                            UCP_DT_REMOTE_SGL_FIELD_LENGTHS);
        sgl.rkeys               = sgl_array(ctx.rkeys, ctx.null_remote_arrays,
                                            UCP_DT_REMOTE_SGL_FIELD_RKEYS);
        return sgl;
    }

    ucp_request_param_t
    make_sgl_param(ucp_dt_remote_sgl_t *remote, size_t remote_count,
                   uint32_t extra_mask = 0) {
        ucp_request_param_t param = {};
        param.op_attr_mask    = UCP_OP_ATTR_FIELD_DATATYPE |
                                UCP_OP_ATTR_FIELD_REMOTE_DATATYPE |
                                UCP_OP_ATTR_FIELD_REMOTE |
                                UCP_OP_ATTR_FIELD_REMOTE_COUNT |
                                extra_mask;
        param.datatype        = ucp_dt_make_sgl();
        param.remote_datatype = ucp_dt_make_sgl();
        param.remote          = remote;
        param.remote_count    = remote_count;
        return param;
    }

    virtual void init();

    /* Factory for the buffers used by @ref test_xfer. Subclasses can override
     * it to change how the buffers are allocated (e.g. asynchronously). */
    virtual mem_buffer *create_mem_buffer(size_t size,
                                          ucs_memory_type_t mem_type)
    {
        return new mem_buffer(size, mem_type);
    }

    void test_xfer(send_func_t send_func, size_t size, unsigned num_iters,
                   size_t alignment, ucs_memory_type_t send_mem_type,
                   ucs_memory_type_t target_mem_type, unsigned mem_map_flags,
                   bool is_ep_flush, bool user_memh, void *arg,
                   size_t reg_offset = 0);
};

#endif
