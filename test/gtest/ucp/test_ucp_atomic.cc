/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

extern "C" {
#include <ucp/core/ucp_types.h> /* for atomic mode */
#include <ucp/core/ucp_mm.h>
}

template <typename T>
class test_ucp_atomic : public test_ucp_memheap {
public:
    /* Test variants */
    enum {
        ATOMIC_MODE  = UCS_MASK(2),
        ENABLE_PROTO = UCS_BIT(2)
    };

    static void get_test_variants(std::vector<ucp_test_variant>& variants,
                                  int variant, const std::string& name)
    {
        uint64_t features = (sizeof(T) == sizeof(uint32_t)) ?
                            UCP_FEATURE_AMO32 : UCP_FEATURE_AMO64;
        add_variant_with_value(variants, features, variant | UCP_ATOMIC_MODE_CPU, "cpu" + name);
        add_variant_with_value(variants, features, variant | UCP_ATOMIC_MODE_DEVICE, "device" + name);
        add_variant_with_value(variants, features, variant | UCP_ATOMIC_MODE_GUESS, "guess" + name);
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        get_test_variants(variants, 0, "");
        get_test_variants(variants, ENABLE_PROTO, "/proto");
    }

    struct send_func_data {
        ucp_atomic_op_t   op;
        ucs_memory_type_t send_mem_type;
        ucs_memory_type_t recv_mem_type;
    };

    void post(size_t size, void *target_ptr, ucp_rkey_h rkey,
              void *expected_data, void *arg)
    {
        const send_func_data* data = (send_func_data*)arg;
        T value                    = (T)ucs::rand() * (T)ucs::rand();
        T prev;
        T result;

        mem_buffer::copy_to(expected_data, &value, sizeof(T),
                            data->send_mem_type);
        mem_buffer::copy_from(&prev, target_ptr, sizeof(T),
                              data->recv_mem_type);
        result = atomic_op_result(data->op, value, prev, 0);

        ucp_request_param_t param;
        param.op_attr_mask = 0;
        ucs_status_t status = do_atomic(data->op, size, target_ptr, rkey,
                                        expected_data, param);
        ASSERT_UCS_OK(status);
        mem_buffer::copy_to(expected_data, &result, sizeof(T),
                            data->send_mem_type);
    }

    void misaligned_post(size_t size, void *target_ptr, ucp_rkey_h rkey,
                         void *expected_data, void *arg)
    {
        const send_func_data* data = (send_func_data*)arg;
        T value = 0;

        /* remote should not change */
        mem_buffer::copy_between(expected_data, target_ptr, sizeof(T),
                                 data->send_mem_type, data->recv_mem_type);

        ucp_request_param_t param;
        param.op_attr_mask  = 0;
        ucs_status_t status = do_atomic(*(ucp_atomic_op_t*)arg, size,
                                        UCS_PTR_BYTE_OFFSET(target_ptr, 1),
                                        rkey, &value, param);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    }

    void fetch(size_t size, void *target_ptr, ucp_rkey_h rkey,
               void *expected_data, void *arg)
    {
        const send_func_data* data = (send_func_data*)arg;
        T value                    = (T)ucs::rand() * (T)ucs::rand();
        T prev;
        T reply_data;
        T result;

        mem_buffer::copy_to(expected_data, &value, sizeof(T),
                            data->send_mem_type);
        mem_buffer::copy_from(&prev, target_ptr, sizeof(T),
                              data->recv_mem_type);
        reply_data = ((data->op == UCP_ATOMIC_OP_CSWAP) && (ucs::rand() % 2)) ?
                     prev : /* cswap success */
                     ((T)ucs::rand() * (T)ucs::rand());
        result = atomic_op_result(data->op, value, prev, reply_data);

        ucp_request_param_t param;
        param.op_attr_mask = UCP_OP_ATTR_FIELD_REPLY_BUFFER;
        param.reply_buffer = &reply_data;

        ucs_status_t status = do_atomic(data->op, size, target_ptr, rkey,
                                        expected_data, param);
        ASSERT_UCS_OK(status);
        mem_buffer::copy_to(expected_data, &result, sizeof(T),
                            data->send_mem_type);

        EXPECT_EQ(prev, reply_data); /* expect the previous value */
    }

protected:
    static const uint64_t POST_ATOMIC_OPS  = UCS_BIT(UCP_ATOMIC_OP_ADD) |
                                             UCS_BIT(UCP_ATOMIC_OP_AND) |
                                             UCS_BIT(UCP_ATOMIC_OP_OR)  |
                                             UCS_BIT(UCP_ATOMIC_OP_XOR);
    static const uint64_t FETCH_ATOMIC_OPS = POST_ATOMIC_OPS             |
                                             UCS_BIT(UCP_ATOMIC_OP_SWAP) |
                                             UCS_BIT(UCP_ATOMIC_OP_CSWAP);

    virtual void init() {
        int atomic_mode = get_variant_value() & ATOMIC_MODE;
        const char *atomic_mode_cfg =
                        (atomic_mode == UCP_ATOMIC_MODE_CPU)    ? "cpu" :
                        (atomic_mode == UCP_ATOMIC_MODE_DEVICE) ? "device" :
                        (atomic_mode == UCP_ATOMIC_MODE_GUESS)  ? "guess" :
                        "";
        modify_config("ATOMIC_MODE", atomic_mode_cfg);

        if (get_variant_value() & ENABLE_PROTO) {
            modify_config("PROTO_ENABLE", "y");
        }

        test_ucp_memheap::init();
    }

    static unsigned default_num_iters() {
        return ucs_max(100 / ucs::test_time_multiplier(), 1);
    }

    void test(send_func_t send_func, uint64_t op_mask,
              unsigned num_iters = default_num_iters()) {
        test_mem_types(send_func, num_iters, op_mask, true);
        test_mem_types(send_func, num_iters, op_mask, false);
    }

private:
    static T atomic_op_result(ucp_atomic_op_t op, T x, T y, T z) {
        switch (op) {
        case UCP_ATOMIC_OP_ADD:
            return x + y;
        case UCP_ATOMIC_OP_SWAP:
            return x;
        case UCP_ATOMIC_OP_CSWAP:
            return (x == y) ? z : y;
        case UCP_ATOMIC_OP_AND:
            return x & y;
        case UCP_ATOMIC_OP_OR:
            return x | y;
        case UCP_ATOMIC_OP_XOR:
            return x ^ y;
        default:
            return 0;
        }
    }

    static std::string opcode_name(ucp_atomic_op_t op) {
        switch (op) {
        case UCP_ATOMIC_OP_ADD:
            return "ADD";
        case UCP_ATOMIC_OP_SWAP:
            return "SWAP";
        case UCP_ATOMIC_OP_CSWAP:
            return "CSWAP";
        case UCP_ATOMIC_OP_AND:
            return "AND";
        case UCP_ATOMIC_OP_OR:
            return "OR";
        case UCP_ATOMIC_OP_XOR:
            return "XOR";
        default:
            return 0;
        }
    }

    ucs_status_t do_atomic(ucp_atomic_op_t op, size_t size, void *target_ptr,
                           ucp_rkey_h rkey, void* value, ucp_request_param_t &param) {

        param.op_attr_mask |= UCP_OP_ATTR_FIELD_DATATYPE;
        param.datatype      = ucp_dt_make_contig(sizeof(T));

        EXPECT_EQ(sizeof(T), size);
        ucs_status_ptr_t status_ptr = ucp_atomic_op_nbx(sender().ep(), op,
                                                        value, 1,
                                                        (uintptr_t)target_ptr,
                                                        rkey, &param);
        return request_wait(status_ptr);
    }

    void test_mem_types(send_func_t send_func, unsigned num_iters,
                        uint64_t op_mask, int is_ep_flush) {
        const int atomic_mode                                     =
                get_variant_value() & ATOMIC_MODE;
        const std::vector<std::vector<ucs_memory_type_t> >& pairs =
                ucs::supported_mem_type_pairs();

        for (size_t i = 0; i < pairs.size(); ++i) {
            ucs_memory_type_t send_mem_type = pairs[i][0], recv_mem_type = pairs[i][1];
            if (!UCP_MEM_IS_HOST(send_mem_type) || !UCP_MEM_IS_HOST(recv_mem_type)) {
                /* Memory type atomics are fully supported only with new protocols */
                if (!(get_variant_value() & ENABLE_PROTO)) {
                    continue;
                }

                static const std::string tls[] = { "ud_v", "ud_x", "rc_v", "tcp" };
                /* Target memory type atomics emulation not supported yet */
                if (((atomic_mode == UCP_ATOMIC_MODE_CPU) ||
                     has_any_transport(tls, ucs_static_array_size(tls))) &&
                    !UCP_MEM_IS_HOST(recv_mem_type)) {
                    continue;
                }

                /* GPU-direct to managed not supported */
                if ((atomic_mode != UCP_ATOMIC_MODE_CPU) &&
                    UCP_MEM_IS_CUDA_MANAGED(recv_mem_type)) {
                    continue;
                }

                /* GPU-direct required for destination CUDA */
                if (UCP_MEM_IS_CUDA(recv_mem_type) &&
                    !check_reg_mem_types(sender(), recv_mem_type)) {
                    continue;
                }
            }

            test_all_opcodes(send_func, num_iters, op_mask, is_ep_flush,
                             send_mem_type, recv_mem_type);
        }
    }

    void test_all_opcodes(send_func_t send_func, unsigned num_iters,
                          uint64_t op_mask, int is_ep_flush,
                          ucs_memory_type_t send_mem_type,
                          ucs_memory_type_t recv_mem_type) {
        ucs::detail::message_stream ms("INFO");

        ms << ucs_memory_type_names[send_mem_type] << "->" <<
              ucs_memory_type_names[recv_mem_type] << " ";

        unsigned op_value;
        ucs_for_each_bit(op_value, op_mask) {
            send_func_data data;
            data.op            = static_cast<ucp_atomic_op_t>(op_value);
            data.send_mem_type = send_mem_type;
            data.recv_mem_type = recv_mem_type;

            ms << opcode_name(data.op) << " ";
            test_xfer(send_func, sizeof(T), num_iters, sizeof(T),
                      send_mem_type, recv_mem_type, 0,
                      is_ep_flush, &data);
        }
    }
};

class test_ucp_atomic32 : public test_ucp_atomic<uint32_t> {
};

UCS_TEST_P(test_ucp_atomic32, post) {
    test(static_cast<send_func_t>(&test_ucp_atomic32::post), POST_ATOMIC_OPS);
}

UCS_TEST_P(test_ucp_atomic32, fetch) {
    test(static_cast<send_func_t>(&test_ucp_atomic32::fetch), FETCH_ATOMIC_OPS);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_atomic32)

class test_ucp_atomic64 : public test_ucp_atomic<uint64_t> {
};

UCS_TEST_P(test_ucp_atomic64, post) {
    test(static_cast<send_func_t>(&test_ucp_atomic64::post), POST_ATOMIC_OPS);
}

UCS_TEST_P(test_ucp_atomic64, fetch) {
    test(static_cast<send_func_t>(&test_ucp_atomic64::fetch), FETCH_ATOMIC_OPS);
}


#if ENABLE_PARAMS_CHECK
UCS_TEST_P(test_ucp_atomic32, misaligned_post) {
    {
        /* Test that unaligned addresses generate error */
        scoped_log_handler slh(hide_errors_logger);
        test(static_cast<send_func_t>(&test_ucp_atomic32::misaligned_post),
             UCS_BIT(UCP_ATOMIC_OP_ADD), 1);
    }
}

UCS_TEST_P(test_ucp_atomic64, misaligned_post) {
    {
        /* Test that unaligned addresses generate error */
        scoped_log_handler slh(hide_errors_logger);
        test(static_cast<send_func_t>(&test_ucp_atomic64::misaligned_post),
             UCS_BIT(UCP_ATOMIC_OP_ADD), 1);
    }
}
#endif

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_atomic64)
