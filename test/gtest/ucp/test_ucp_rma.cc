/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
#include <ucs/sys/sys.h>


class test_ucp_rma : public test_ucp_memheap {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }

    std::vector<ucp_test_param>
    static enum_test_params(const ucp_params_t& ctx_params,
                            const std::string& name,
                            const std::string& test_case_name,
                            const std::string& tls)
    {
        std::vector<ucp_test_param> result;
        generate_test_params_variant(ctx_params, name, test_case_name, tls, 0,
                                     result);
        generate_test_params_variant(ctx_params, name, test_case_name + "/map_nb",
                                     tls, UCP_MEM_MAP_NONBLOCK, result);
        return result;
    }

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

        ucs::fill_random(memheap_addr, ucs_min(max_size, 16384U));
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

        ucs::fill_random(memheap_addr, ucs_min(max_size, 16384U));
        status = ucp_get(e->ep(), (void *)&expected_data[0], expected_data.length(),
                         (uintptr_t)memheap_addr, rkey);
        ASSERT_UCS_OK(status);
    }

    void test_message_sizes(blocking_send_func_t func, size_t *msizes, int iters, int is_nbi);
};

void test_ucp_rma::test_message_sizes(blocking_send_func_t func, size_t *msizes, int iters, int is_nbi)
{
   int i;

   for (i = 0; msizes[i] > 0; i++) {
       if (is_nbi) {
           test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(func),
                                                 msizes[i], i, 1, false, false);
       } else {
           test_blocking_xfer(func, msizes[i], iters, 1, false, false);
       }
   }
}

UCS_TEST_P(test_ucp_rma, blocking_small) {
    size_t sizes[] = { 8, 24, 96, 120, 250, 0};

    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       sizes, 1000, 0);
    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get), 
                       sizes, 1000, 0);
}

UCS_TEST_P(test_ucp_rma, blocking_med) {
    size_t sizes[] = { 1000, 3000, 9000, 17300, 31000, 99000, 130000, 0};

    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       sizes, 100, 0);
    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get), 
                       sizes, 100, 0);
}

static const size_t MEG = 1024 * 1024ULL;

UCS_TEST_P(test_ucp_rma, blocking_large) {
    size_t sizes[] = { 1 * MEG, 3 * MEG, 9 * MEG, 17 * MEG, 32 * MEG, 0};

    if (RUNNING_ON_VALGRIND) {
        UCS_TEST_SKIP_R("skipping on valgrind");
    }
    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       sizes, 3, 0);
    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get), 
                       sizes, 3, 0);
}

UCS_TEST_P(test_ucp_rma, nbi_small) {
    size_t sizes[] = { 8, 24, 96, 120, 250, 0};

    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       sizes, 1000, 1);
    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi), 
                       sizes, 1000, 1);
}

UCS_TEST_P(test_ucp_rma, nbi_med) {
    size_t sizes[] = { 1000, 3000, 9000, 17300, 31000, 99000, 130000, 0};

    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       sizes, 100, 1);
    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi), 
                       sizes, 100, 1);
}

UCS_TEST_P(test_ucp_rma, nbi_large) {
    size_t sizes[] = { 1 * MEG, 3 * MEG, 9 * MEG, 17 * MEG, 32 * MEG, 0};

    if (RUNNING_ON_VALGRIND) {
        UCS_TEST_SKIP_R("skipping on valgrind");
    }

    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       sizes, 3, 1);
    test_message_sizes(static_cast<blocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi), 
                       sizes, 3, 1);
}

UCS_TEST_P(test_ucp_rma, blocking_put_allocated) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, false);
}

UCS_TEST_P(test_ucp_rma, blocking_put_registered) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_put),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_put_nbi_flush_worker) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, false);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_put_nbi_flush_ep) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, true);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, true);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_put_nbi_flush_worker) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, false);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_put_nbi_flush_ep) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, true);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_put_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, true);
}

UCS_TEST_P(test_ucp_rma, blocking_get) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, false);
    test_blocking_xfer(static_cast<blocking_send_func_t>(&test_ucp_rma::blocking_get),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_get_nbi_flush_worker) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, false);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_get_nbi_flush_ep) {
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, true);
    test_blocking_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, true);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_get_nbi_flush_worker) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, false);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, false);
}

UCS_TEST_P(test_ucp_rma, nonblocking_stream_get_nbi_flush_ep) {
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, false, true);
    test_nonblocking_implicit_stream_xfer(static_cast<nonblocking_send_func_t>(&test_ucp_rma::nonblocking_get_nbi),
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       1, true, true);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_rma)
