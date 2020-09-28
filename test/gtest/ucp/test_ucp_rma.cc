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
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_RMA, 0, "flush_worker");
        add_variant_with_value(variants, UCP_FEATURE_RMA, FLUSH_EP, "flush_ep");
        add_variant_with_value(variants, UCP_FEATURE_RMA,
                               FLUSH_EP | ENABLE_PROTO, "flush_ep_proto");
    }

    virtual void init() {
        if (enable_proto()) {
            modify_config("PROTO_ENABLE", "y");
        }
        test_ucp_memheap::init();
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
        ucs_status_ptr_t status_ptr = do_get(size, target_ptr, rkey,
                                             expected_data);
        request_wait(status_ptr);
    }

    void get_nbi(size_t size, void *target_ptr, ucp_rkey_h rkey,
                 void *expected_data, void *arg) {
        ucs_status_ptr_t status_ptr = do_get(size, target_ptr, rkey,
                                             expected_data);
        request_release(status_ptr);
    }

protected:
    void test_mem_types(send_func_t send_func, bool proto_supports_memtype) {
        std::vector<std::vector<ucs_memory_type_t> > pairs =
                ucs::supported_mem_type_pairs();

        for (size_t i = 0; i < pairs.size(); ++i) {

            // If memtype supporting protocols are not enabled, and memory type
            // is not accessible from CPU, skip the test.
            // TODO remove this check after memory types is fully supported by
            // RMA API
            if (!(enable_proto() && proto_supports_memtype) &&
                (!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(pairs[i][0]) ||
                 !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(pairs[i][1]))) {
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
        FLUSH_EP     = UCS_BIT(0), /* If not set, flush worker */
        ENABLE_PROTO = UCS_BIT(1)
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

    ucs_status_ptr_t do_get(size_t size, void *target_ptr, ucp_rkey_h rkey,
                            void *expected_data) {
        ucp_request_param_t param;

        param.op_attr_mask = 0;
        return ucp_get_nbx(sender().ep(), expected_data, size,
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

            ucs_memory_type_t mem_types[] = {send_mem_type, target_mem_type};
            test_xfer(send_func, size, num_iters, 1, send_mem_type,
                      target_mem_type, mem_map_flags, is_ep_flush(), mem_types);
       }
    }

    bool is_ep_flush() {
        return get_variant_value() & FLUSH_EP;
    }

    bool enable_proto() {
        return get_variant_value() & ENABLE_PROTO;
    }
};

UCS_TEST_P(test_ucp_rma, put_blocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::put_b), true);
}

UCS_TEST_P(test_ucp_rma, put_nonblocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::put_nbi), true);
}

UCS_TEST_P(test_ucp_rma, get_blocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_b), false);
}

UCS_TEST_P(test_ucp_rma, get_nonblocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_nbi), false);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_rma)
