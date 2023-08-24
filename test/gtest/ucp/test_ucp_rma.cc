/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h> /* for UCP_MEM_IS_ACCESSIBLE_FROM_CPU */
#include <ucs/sys/sys.h>
}


class test_ucp_rma : public test_ucp_memheap {
public:
    using iov_op_t = ucs_status_ptr_t (test_ucp_rma::*)(
            size_t size, void *expected_data, ucp_request_param_t *param,
            void *target_ptr, ucp_rkey_h rkey, ucp_dt_iov_t *iov,
            size_t iov_count, void *arg);

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_RMA, 0, "flush_worker");
        add_variant_with_value(variants, UCP_FEATURE_RMA, FLUSH_EP, "flush_ep");
        if (!RUNNING_ON_VALGRIND) {
            add_variant_with_value(variants, UCP_FEATURE_RMA,
                                   FLUSH_EP | DISABLE_PROTO,
                                   "flush_ep_proto_v1");
        }
        add_variant_with_value(variants, UCP_FEATURE_RMA, USER_MEMH,
                               "user_memh");
    }

    virtual void init() {
        if (disable_proto()) {
            modify_config("PROTO_ENABLE", "n");
        } else {
            modify_config("MAX_RMA_LANES", "2");
        }
        test_ucp_memheap::init();
    }

    void do_nbi_iov(iov_op_t op, size_t size, void *expected_data,
                    void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        ucp_dt_iov_t iov[UCP_MAX_IOV];
        ucs_status_ptr_t status_ptr;
        ucp_request_param_t param;

        param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        param.datatype     = DATATYPE_IOV;

        for (auto iov_count = 0; iov_count <= UCP_MAX_IOV;
             iov_count += ucs_max(iov_count, 1)) {
            status_ptr = (this->*op)(size, expected_data, &param, target_ptr,
                                     rkey, iov, iov_count, arg);
            flush_ep(sender());
            request_release(status_ptr);
        }
    }

    void put_b(size_t size, void *expected_data, ucp_mem_h memh,
               void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        ucs_status_ptr_t status_ptr = do_put(size, expected_data, memh,
                                             target_ptr, rkey, arg);
        request_wait(status_ptr);
    }

    void put_nbi(size_t size, void *expected_data, ucp_mem_h memh,
                 void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        ucs_status_ptr_t status_ptr = do_put(size, expected_data, memh,
                                             target_ptr, rkey, arg);
        request_release(status_ptr);
    }

    void put_nbi_iov(size_t size, void *expected_data, ucp_mem_h memh,
                     void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        do_nbi_iov(&test_ucp_rma::do_put_iov, size, expected_data, target_ptr,
                   rkey, arg);
    }

    void get_b(size_t size, void *expected_data, ucp_mem_h memh,
               void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        ucs_status_ptr_t status_ptr = do_get(size, expected_data, memh,
                                             target_ptr, rkey);
        request_wait(status_ptr);
    }

    void get_nbi(size_t size, void *expected_data, ucp_mem_h memh,
                 void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        ucs_status_ptr_t status_ptr = do_get(size, expected_data, memh,
                                             target_ptr, rkey);
        request_release(status_ptr);
    }

    void get_nbi_iov(size_t size, void *expected_data, ucp_mem_h memh,
                     void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        do_nbi_iov(&test_ucp_rma::do_get_iov, size, expected_data, target_ptr,
                   rkey, arg);
    }

protected:
    static size_t default_max_size() {
        return (100 * UCS_MBYTE) / ucs::test_time_multiplier();
    }

    void test_mem_types(send_func_t send_func, size_t min_size = 128,
                        size_t max_size = default_max_size()) {
        const std::vector<std::vector<ucs_memory_type_t> >& pairs =
                ucs::supported_mem_type_pairs();

        for (size_t i = 0; i < pairs.size(); ++i) {

            /* Memory type put/get is fully supported only with new protocols */
            if (!m_ucp_config->ctx.proto_enable &&
                (!UCP_MEM_IS_HOST(pairs[i][0]) ||
                 !UCP_MEM_IS_HOST(pairs[i][1]))) {
                continue;
            }

            test_message_sizes(send_func, min_size, max_size, pairs[i][0],
                               pairs[i][1], 0);
        }

        /* test non-blocking map with host memory */
        test_message_sizes(send_func, min_size, max_size, UCS_MEMORY_TYPE_HOST,
                           UCS_MEMORY_TYPE_HOST, UCP_MEM_MAP_NONBLOCK);
    }

    void test_message_sizes(send_func_t send_func) {
        test_message_sizes(send_func, 128, default_max_size(),
                           UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_HOST, 0);
    }

    bool disable_proto() {
        return !m_ucp_config->ctx.proto_enable;
    }

    bool user_memh()
    {
        return get_variant_value() & USER_MEMH;
    }

private:
    /* Test variants */
    enum {
        FLUSH_EP      = UCS_BIT(0), /* If not set, flush worker */
        DISABLE_PROTO = UCS_BIT(1),
        USER_MEMH     = UCS_BIT(2),
    };

    void init_iov(size_t size, ucp_dt_iov_t *iov, size_t iov_count,
                  void *expected_data)  {
        const auto iov_buffer_length = size / iov_count;

        for (size_t i = 0; i < iov_count; ++i) {
            iov[i].buffer = UCS_PTR_BYTE_OFFSET(expected_data,
                                                i * iov_buffer_length);
            iov[i].length = iov_buffer_length;
        }

        iov[iov_count - 1].length += size % iov_count;
    }

    void request_param_init(ucp_request_param_t *param, ucp_mem_h memh)
    {
        param->op_attr_mask = 0;

        if (memh == NULL) {
            return;
        }

        param->op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
        param->memh          = memh;
    }

    ucs_status_ptr_t do_put(size_t size, void *expected_data, ucp_mem_h memh,
                            void *target_ptr, ucp_rkey_h rkey, void *arg)
    {
        ucs_memory_type_t *mem_types = reinterpret_cast<ucs_memory_type_t*>(arg);
        mem_buffer::pattern_fill(expected_data, size, ucs::rand(), mem_types[0]);

        ucp_request_param_t param;
        request_param_init(&param, memh);

        return ucp_put_nbx(sender().ep(), expected_data, size,
                           (uintptr_t)target_ptr, rkey, &param);
    }

    ucs_status_ptr_t do_put_iov(size_t size, void *expected_data,
                                ucp_request_param_t *param, void *target_ptr,
                                ucp_rkey_h rkey, ucp_dt_iov_t *iov,
                                size_t iov_count, void *arg)
    {
        ucs_memory_type_t *mem_types;

        if (iov_count > 0) {
            mem_types = reinterpret_cast<ucs_memory_type_t*>(arg);
            mem_buffer::pattern_fill(expected_data, size, ucs::rand(),
                                     mem_types[0]);

            init_iov(size, iov, iov_count, expected_data);
        }

        return ucp_put_nbx(sender().ep(), iov, iov_count, (uintptr_t)target_ptr,
                           rkey, param);
    }

    ucs_status_ptr_t do_get(size_t size, void *expected_data, ucp_mem_h memh,
                            void *target_ptr, ucp_rkey_h rkey)
    {
        ucp_request_param_t param;
        request_param_init(&param, memh);

        return ucp_get_nbx(sender().ep(), expected_data, size,
                           (uintptr_t)target_ptr, rkey, &param);
    }

    ucs_status_ptr_t do_get_iov(size_t size, void *expected_data,
                                ucp_request_param_t *param, void *target_ptr,
                                ucp_rkey_h rkey, ucp_dt_iov_t *iov,
                                size_t iov_count, void * /*arg*/)
    {
        if (iov_count > 0) {
            init_iov(size, iov, iov_count, expected_data);
        }

        return ucp_get_nbx(sender().ep(), iov, iov_count, (uintptr_t)target_ptr,
                           rkey, param);
    }

    void test_message_sizes(send_func_t send_func, size_t min_size,
                            size_t max_size, ucs_memory_type_t send_mem_type,
                            ucs_memory_type_t target_mem_type,
                            unsigned mem_map_flags) {
        ucs::detail::message_stream ms("INFO");

        ucs_assert(min_size <= max_size);
        ms << ucs_memory_type_names[send_mem_type] << "->" <<
              ucs_memory_type_names[target_mem_type] << " ";
        if (mem_map_flags & UCP_MEM_MAP_NONBLOCK) {
            ms << "map_nb ";
        }

        /* Test different random sizes */
        for (size_t current_max_size = min_size; current_max_size <= max_size;
             current_max_size *= 4) {
            size_t size = min_size;
            if (min_size < current_max_size) {
                size += ucs::rand() % (current_max_size - min_size);
            }

            unsigned num_iters = ucs_min(100, max_size / (size + 1));
            num_iters          = ucs_max(1, num_iters / ucs::test_time_multiplier());

            ms << num_iters << "x" << size << " ";
            fflush(stdout);

            ucs_memory_type_t mem_types[] = {send_mem_type, target_mem_type};
            test_xfer(send_func, size, num_iters, 1, send_mem_type,
                      target_mem_type, mem_map_flags, is_ep_flush(), user_memh(),
                      mem_types);

            if (HasFailure() || (num_errors() > 0)) {
                break;
            }
        }
    }

    bool is_ep_flush() {
        return get_variant_value() & FLUSH_EP;
    }

};

class test_ucp_rma_reg_nb : public test_ucp_rma {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_RMA, 0, "");
        /* TODO pending MOFED 23.07 upgrade
        add_variant_with_value(variants, UCP_FEATURE_RMA, NO_DEVX, "no_devx");
        */
    }

    virtual void init() {
        modify_config("REG_NONBLOCK_MEM_TYPES", "host");
        if (get_variant_value() & NO_DEVX) {
            modify_config("IB_MLX5_DEVX_OBJECTS", "", SETENV_IF_NOT_EXIST);
        }
        test_ucp_rma::init();
    }

private:
    enum {
        NO_DEVX = UCS_BIT(0),
    };
};

UCS_TEST_P(test_ucp_rma, put_blocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::put_b));
}

UCS_TEST_P(test_ucp_rma, put_nonblocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::put_nbi));
}

UCS_TEST_SKIP_COND_P(test_ucp_rma, put_nonblocking_iov_zcopy, disable_proto(),
                     "ZCOPY_THRESH=0") {
    if (!sender().has_lane_with_caps(UCT_IFACE_FLAG_PUT_ZCOPY)) {
        UCS_TEST_SKIP_R("put_zcopy is not supported");
    }

    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::put_nbi_iov), 2000,
                   64 * UCS_KBYTE);
}


UCS_TEST_P(test_ucp_rma, get_blocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_b));
}

UCS_TEST_P(test_ucp_rma, get_nonblocking) {
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_nbi));
}

UCS_TEST_SKIP_COND_P(test_ucp_rma, get_nonblocking_iov_zcopy, disable_proto(),
                     "ZCOPY_THRESH=0") {
    if (!sender().has_lane_with_caps(UCT_IFACE_FLAG_GET_ZCOPY)) {
        UCS_TEST_SKIP_R("get_zcopy is not supported");
    }

    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_nbi_iov), 2000,
                   64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_rma, get_blocking_zcopy, "ZCOPY_THRESH=0") {
    /* test get_zcopy minimal message length is respected */
    test_mem_types(static_cast<send_func_t>(&test_ucp_rma::get_b), 128,
                   64 * UCS_KBYTE);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_rma)

UCS_TEST_P(test_ucp_rma_reg_nb, put_blocking) {
    test_message_sizes(static_cast<send_func_t>(&test_ucp_rma::put_b));
}

UCS_TEST_P(test_ucp_rma_reg_nb, put_nonblocking) {
    test_message_sizes(static_cast<send_func_t>(&test_ucp_rma::put_nbi));
}

UCS_TEST_P(test_ucp_rma_reg_nb, get_blocking) {
    test_message_sizes(static_cast<send_func_t>(&test_ucp_rma::get_b));
}

UCS_TEST_P(test_ucp_rma_reg_nb, get_nonblocking) {
    test_message_sizes(static_cast<send_func_t>(&test_ucp_rma::get_nbi));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_rma_reg_nb)
