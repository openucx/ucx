/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

#include <ucs/sys/sys.h>
extern "C" {
#include <ucp/core/ucp_mm.h> /* for UCP_MEM_IS_ACCESSIBLE_FROM_CPU */
}


class test_ucp_rma : public test_ucp_memheap {
private:
    static void send_completion(void *request, ucs_status_t status){}
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params, const std::string& name,
                     const std::string& test_case_name, const std::string& tls) {
        std::vector<ucp_test_param> result;
        generate_test_params_variant(ctx_params, name, test_case_name + "/flush_ep",
                                     tls, FLUSH_EP, result);
        generate_test_params_variant(ctx_params, name, test_case_name + "/flush_worker",
                                     tls, FLUSH_WORKER, result);
        return result;
    }

    void put_b(size_t size, void *target_ptr, ucp_rkey_h rkey,
               void *expected_data, void *arg) {
        ucs_status_ptr_t status_ptr = do_put(size, target_ptr, rkey,
                                             expected_data, arg);
        request_wait(status_ptr);
    }

    void put_nbi(size_t size, void *target_ptr, ucp_rkey_h rkey,
                 void *expected_data, void *arg) {
        ucs_status_ptr_t status_ptr = do_put(size, target_ptr, rkey,
                                             expected_data, arg);
        request_release(status_ptr);
    }

    void get_b(size_t size, void *target_ptr, ucp_rkey_h rkey,
               void *expected_data, void *arg) {
        ucs_status_ptr_t status_ptr;
        ucp_request_param_t param;

        param.op_attr_mask = 0;
        status_ptr = ucp_get_nbx(sender().ep(), expected_data, size,
                                 (uintptr_t)target_ptr, rkey, &param);
        request_wait(status_ptr);
    }

    void get_nbi(size_t size, void *target_ptr, ucp_rkey_h rkey,
                 void *expected_data, void *arg) {
        ucs_status_ptr_t status_ptr;
        ucp_request_param_t param;

        param.op_attr_mask = 0;
        status_ptr = ucp_get_nbx(sender().ep(), expected_data, size,
                                 (uintptr_t)target_ptr, rkey, &param);
        request_release(status_ptr);
    }

protected:
    void test_mem_types(send_func_t send_func) {
        std::vector<std::vector<ucs_memory_type_t> > pairs =
                ucs::supported_mem_type_pairs();

        for (size_t i = 0; i < pairs.size(); ++i) {

            // TODO remove this check after memory types is fully supported by
            // RMA API
            if (!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(pairs[i][0]) ||
                !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(pairs[i][1])) {
                continue;
            }

            test_message_sizes(send_func, pairs[i][0], pairs[i][1], 0);
        }

        /* test non-blocking map with host memory */
        test_message_sizes(send_func, UCS_MEMORY_TYPE_HOST,
                           UCS_MEMORY_TYPE_HOST, UCP_MEM_MAP_NONBLOCK);
    }

private:
    /* Test variants */
    enum {
        FLUSH_EP,
        FLUSH_WORKER
    };

    ucs_status_ptr_t do_put(size_t size, void *target_ptr, ucp_rkey_h rkey,
                            void *expected_data, void *arg) {
        ucs_memory_type_t *mem_types = reinterpret_cast<ucs_memory_type_t*>(arg);
        mem_buffer::pattern_fill(expected_data, size, ucs::rand(), mem_types[0]);

        ucp_request_param_t param;
        param.op_attr_mask = 0;
        return ucp_put_nbx(sender().ep(), expected_data, size,
                           (uintptr_t)target_ptr, rkey, &param);
    }

    void test_message_sizes(send_func_t send_func,
                            ucs_memory_type_t send_mem_type,
                            ucs_memory_type_t target_mem_type,
                            unsigned mem_map_flags) {
        static const size_t MAX_SIZE = (100 * UCS_MBYTE) /
                                       ucs::test_time_multiplier();
        ucs::detail::message_stream ms("INFO");

        ms << ucs_memory_type_names[send_mem_type] << "->" <<
              ucs_memory_type_names[target_mem_type] << " ";
        if (mem_map_flags & UCP_MEM_MAP_NONBLOCK) {
            ms << "map_nb ";
        }

        /* Test different random sizes */
        for (size_t current_max_size = 128; current_max_size < MAX_SIZE;
             current_max_size *= 4) {

            size_t size        = ucs::rand() % current_max_size;
            unsigned num_iters = ucs_min(100, MAX_SIZE / (size + 1));
            num_iters          = ucs_max(1, num_iters / ucs::test_time_multiplier());

            ms << num_iters << "x" << size << " ";
            fflush(stdout);

            ucs_memory_type_t mem_types[2] = {send_mem_type, target_mem_type};
            test_xfer(send_func, size, num_iters, 1, send_mem_type,
                      target_mem_type, mem_map_flags, is_ep_flush(), mem_types);
       }
    }

    bool is_ep_flush() {
        return GetParam().variant == FLUSH_EP;
    }
};

UCS_TEST_P(test_ucp_rma, put_blocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::put_b));
}

UCS_TEST_P(test_ucp_rma, put_nonblocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::put_nbi));
}

UCS_TEST_P(test_ucp_rma, get_blocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_b));
}

UCS_TEST_P(test_ucp_rma, get_nonblocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_nbi));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_rma)
