/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include <common/mem_buffer.h>

extern "C" {
#include <uct/api/uct.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
}


#define UCP_INSTANTIATE_TEST_CASE_MEMTYPE(_test_case, _name, _mem_type) \
    INSTANTIATE_TEST_CASE_P(_name, _test_case, \
                            testing::ValuesIn(_test_case::enum_test_params( \
                                              _test_case::get_ctx_params(), \
                                              #_test_case, _mem_type)));

#define UCP_INSTANTIATE_TEST_CASE_MEMTYPES(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_MEMTYPE(_test_case, host,         UCS_MEMORY_TYPE_HOST) \
    UCP_INSTANTIATE_TEST_CASE_MEMTYPE(_test_case, cuda,         UCS_MEMORY_TYPE_CUDA) \
    UCP_INSTANTIATE_TEST_CASE_MEMTYPE(_test_case, cuda_managed, UCS_MEMORY_TYPE_CUDA_MANAGED) \
    UCP_INSTANTIATE_TEST_CASE_MEMTYPE(_test_case, rocm,         UCS_MEMORY_TYPE_ROCM) \
    UCP_INSTANTIATE_TEST_CASE_MEMTYPE(_test_case, rocm_managed, UCS_MEMORY_TYPE_ROCM_MANAGED)

class test_ucp_mem_type : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_TAG;
        return params;
    }

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& test_case_name, ucs_memory_type_t mem_type)
    {
        std::vector<ucp_test_param> result;

        std::vector<ucs_memory_type_t> mem_types =
                        mem_buffer::supported_mem_types();
        if (std::find(mem_types.begin(), mem_types.end(), mem_type) !=
            mem_types.end()) {
            generate_test_params_variant(ctx_params, "all", test_case_name,
                                         "all", mem_type, result);
        }

        return result;
    }

protected:
    ucs_memory_type_t mem_type() const {
        return static_cast<ucs_memory_type_t>(GetParam().variant);
    }
};

UCS_TEST_P(test_ucp_mem_type, detect) {

    const size_t size                      = 256;
    const ucs_memory_type_t alloc_mem_type = mem_type();

    mem_buffer b(size, alloc_mem_type);

    ucs_memory_type_t detected_mem_type =
                    ucp_memory_type_detect(sender().ucph(), b.ptr(), size);
    EXPECT_EQ(alloc_mem_type, detected_mem_type);
}

UCP_INSTANTIATE_TEST_CASE_MEMTYPES(test_ucp_mem_type)

class test_ucp_mem_type_alloc_before_init : public test_ucp_mem_type {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features    |= UCP_FEATURE_TAG;
        return params;
    }

    test_ucp_mem_type_alloc_before_init() {
        m_size = 10000;
    }

    virtual void init() {
        m_send_buffer.reset(new mem_buffer(m_size, mem_type()));
        m_recv_buffer.reset(new mem_buffer(m_size, mem_type()));
        test_ucp_mem_type::init();
    }

    virtual void cleanup() {
        test_ucp_mem_type::cleanup();
        m_send_buffer.reset();
        m_recv_buffer.reset();
    }

    static const uint64_t SEED = 0x1111111111111111lu;
protected:
    size_t                     m_size;
    ucs::auto_ptr<mem_buffer>  m_send_buffer, m_recv_buffer;
};

UCS_TEST_P(test_ucp_mem_type_alloc_before_init, xfer) {
    sender().connect(&receiver(), get_ep_params());

    EXPECT_EQ(mem_type(), ucp_memory_type_detect(sender().ucph(),
                                                 m_send_buffer->ptr(), m_size));
    EXPECT_EQ(mem_type(), ucp_memory_type_detect(receiver().ucph(),
                                                 m_recv_buffer->ptr(), m_size));

    mem_buffer::pattern_fill(m_send_buffer->ptr(), m_size, SEED, mem_type());

    for (int i = 0; i < 3; ++i) {
        mem_buffer::pattern_fill(m_recv_buffer->ptr(), m_size, 0, mem_type());

        void *sreq = ucp_tag_send_nb(sender().ep(), m_send_buffer->ptr(), m_size,
                                     ucp_dt_make_contig(1), 1,
                                     (ucp_send_callback_t)ucs_empty_function);
        void *rreq = ucp_tag_recv_nb(receiver().worker(), m_recv_buffer->ptr(),
                                     m_size, ucp_dt_make_contig(1), 1, 1,
                                     (ucp_tag_recv_callback_t)ucs_empty_function);
        wait(sreq);
        wait(rreq);

        mem_buffer::pattern_check(m_recv_buffer->ptr(), m_size, SEED, mem_type());
    }
}

UCP_INSTANTIATE_TEST_CASE_MEMTYPES(test_ucp_mem_type_alloc_before_init)
