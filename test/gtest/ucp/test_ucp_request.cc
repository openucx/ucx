/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <string.h>
#include "ucp_test.h"
#include <common/mem_buffer.h>
extern "C" {
#include <ucp/core/ucp_worker.h>
}


class test_ucp_request : public ucp_test {
public:
    virtual void init()
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        int mem_type_pair_index = get_variant_value() %
                                  mem_buffer::supported_mem_types().size();
        mem_type = mem_buffer::supported_mem_types()[mem_type_pair_index];
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        int count = 0;
        add_variant(variants, UCP_FEATURE_TAG);
        std::vector<ucs_memory_type_t>::const_iterator iter;
        std::vector<ucs_memory_type_t> supported_mem_types =
            mem_buffer::supported_mem_types();

        for (iter = supported_mem_types.begin();
             iter != supported_mem_types.end(); ++iter) {
            std::string name = ucs_memory_type_names[*iter];
            add_variant_with_value(variants, UCP_FEATURE_TAG, count, name);
            ++count;
        }
    }

    static const size_t msg_size = 4;

protected:
    ucs_memory_type_t mem_type;
};


UCS_TEST_P(test_ucp_request, test_request_query)
{
    ucp_request_param_t param;
    ucp_request_attr_t attr;
    ucp_worker_attr_t worker_attr;
    void *reqs[2];

    mem_buffer m_recv_mem_buf(msg_size, mem_type);
    mem_buffer m_send_mem_buf(msg_size, mem_type);

    param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;

    void *sreq = ucp_tag_send_nbx(sender().ep(), m_send_mem_buf.ptr(), msg_size,
                                  0, &param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(sreq));
    reqs[0] = sreq;

    void *rreq = ucp_tag_recv_nbx(receiver().worker(), m_recv_mem_buf.ptr(),
                                  msg_size, 0, 0, &param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(rreq));
    reqs[1] = rreq;

    while ((ucp_request_check_status(sreq) == UCS_INPROGRESS) ||
           (ucp_request_check_status(rreq) == UCS_INPROGRESS)) {
        progress();
    }

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_INFO_STRING;
    ucp_worker_query(receiver().worker(), &worker_attr);

    char debug_string[worker_attr.max_debug_string];
    memset(&debug_string, 0, worker_attr.max_debug_string);

    attr.field_mask        = UCP_REQUEST_ATTR_FIELD_INFO_STRING      |
                             UCP_REQUEST_ATTR_FIELD_INFO_STRING_SIZE |
                             UCP_REQUEST_ATTR_FIELD_MEM_TYPE         |
                             UCP_REQUEST_ATTR_FIELD_STATUS;
    attr.debug_string      = debug_string;
    attr.debug_string_size = worker_attr.max_debug_string;

    for (int i = 0; i < 2; i++) {
        const char *req_type = (i == 0) ? "send" : "recv";
        ucp_request_query(UCS_STATUS_PTR(reqs[i]), &attr);
        UCS_TEST_MESSAGE << req_type << " req: " << attr.debug_string;
        std::string str(attr.debug_string);
        EXPECT_GT(str.size(), 0);
        EXPECT_NE(str.find(req_type), std::string::npos);
        EXPECT_NE(str.find(ucs_memory_type_names[mem_type]), std::string::npos);
        ASSERT_EQ(attr.status, UCS_OK);
        ASSERT_EQ(attr.mem_type, mem_type);

        ucp_request_free(reqs[i]);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_request, all, "all")
