/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"


class test_ucp_rma : public test_ucp_memheap {
public:
    using test_ucp_memheap::get_ctx_params;

    void nonblocking_put_nbi(entity *e, size_t max_size,
                             void *memheap_addr,
                             ucp_rkey_h rkey,
                             std::string& expected_data)
    {
        ucs_status_t status;
        status = ucp_put_nbi(e->ep(), &expected_data[0], expected_data.length(),
                             (uintptr_t)memheap_addr, rkey);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
    }

    void blocking_put(entity *e, size_t max_size,
                      void *memheap_addr,
                      ucp_rkey_h rkey,
                      std::string& expected_data)
    {
        ucs_status_t status;
        status = ucp_put(e->ep(), &expected_data[0], expected_data.length(),
                         (uintptr_t)memheap_addr, rkey);
        ASSERT_UCS_OK(status);
    }

    void nonblocking_get_nbi(entity *e, size_t max_size,
                             void *memheap_addr,
                             ucp_rkey_h rkey,
                             std::string& expected_data)
    {
        ucs_status_t status;

        ucs::fill_random((char*)memheap_addr, (char*)memheap_addr + max_size);
        status = ucp_get_nbi(e->ep(), (void *)&expected_data[0], expected_data.length(),
                             (uintptr_t)memheap_addr, rkey);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
    }

    void blocking_get(entity *e, size_t max_size,
                      void *memheap_addr,
                      ucp_rkey_h rkey,
                      std::string& expected_data)
    {
        ucs_status_t status;

        ucs::fill_random((char*)memheap_addr, (char*)memheap_addr + max_size);
        status = ucp_get(e->ep(), (void *)&expected_data[0], expected_data.length(),
                         (uintptr_t)memheap_addr, rkey);
        ASSERT_UCS_OK(status);
    }
};


UCS_TEST_P(test_ucp_rma, blocking_put_allocated) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       1, false, false);
}

UCS_TEST_P(test_ucp_rma, blocking_put_registered) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_put_nbi_flush_worker) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, false, false);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_put_nbi_flush_ep) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, false, true);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, true, true);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_put_nbi_flush_worker) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, false, false);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_put_nbi_flush_ep) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, false, true);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       1, true, true);
}

UCS_TEST_P(test_ucp_rma, blocking_get) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get),
                       1, false, false);
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get),
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_get_nbi_flush_worker) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, false, false);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_get_nbi_flush_ep) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, false, true);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, true, true);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_get_nbi_flush_worker) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, false, false);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_get_nbi_flush_ep) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, false, true);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       1, true, true);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_rma)

