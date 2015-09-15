/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"


class test_ucp_rma : public test_ucp_memheap {
public:
    void blocking_put(entity *e, size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        std::string send_data(max_size, 0);
        ucs::fill_random(send_data.begin(), send_data.end());
        status = ucp_put(e->ep(), &send_data[0], send_data.length(),
                         (uintptr_t)memheap_addr, rkey);
        expected_data = send_data;
        std::fill(send_data.begin(), send_data.end(), 0);
        ASSERT_UCS_OK(status);
    }

    void blocking_get(entity *e, size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        std::string reply_buffer;

        ucs::fill_random((char*)memheap_addr, (char*)memheap_addr + max_size);
        reply_buffer.resize(max_size);
        status = ucp_get(e->ep(), &reply_buffer[0], reply_buffer.length(),
                         (uintptr_t)memheap_addr, rkey);
        expected_data.clear();
        ASSERT_UCS_OK(status);

        EXPECT_EQ(std::string((char*)memheap_addr, reply_buffer.length()),
                  reply_buffer);
    }

    virtual uint64_t features() const {
        return UCP_FEATURE_RMA;
    }
};


UCS_TEST_F(test_ucp_rma, blocking_put) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       1, 0);
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       1, 1);
}

UCS_TEST_F(test_ucp_rma, blocking_get) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get),
                       1, 0);
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get),
                       1, 1);
}
