/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

extern "C" {
#include <ucp/core/ucp_types.h> /* for atomic mode */
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

    void post(size_t size, void *target_ptr, ucp_rkey_h rkey,
              void *expected_data, void *arg)
    {
        ucp_atomic_op_t op = *(ucp_atomic_op_t*)arg;
        T value            = (T)ucs::rand() * (T)ucs::rand();
        T prev             = *(T*)target_ptr;
        *(T*)expected_data = atomic_op_result(op, value, prev, 0);

        ucp_request_param_t param;
        param.op_attr_mask = 0;
        ucs_status_t status = do_atomic(op, size, target_ptr, rkey, value, param);
        ASSERT_UCS_OK(status);
    }

    void misaligned_post(size_t size, void *target_ptr, ucp_rkey_h rkey,
                         void *expected_data, void *arg)
    {
        *(T*)expected_data = *(T*)target_ptr; /* remote should not change */

        ucp_request_param_t param;
        param.op_attr_mask  = 0;
        ucs_status_t status = do_atomic(*(ucp_atomic_op_t*)arg, size,
                                        UCS_PTR_BYTE_OFFSET(target_ptr, 1),
                                        rkey, 0, param);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    }

    void fetch(size_t size, void *target_ptr, ucp_rkey_h rkey,
               void *expected_data, void *arg)
    {
        ucp_atomic_op_t op = *(ucp_atomic_op_t*)arg;
        T value            = (T)ucs::rand() * (T)ucs::rand();
        T prev             = *(T*)target_ptr;
        T reply_data       = ((op == UCP_ATOMIC_OP_CSWAP) && (ucs::rand() % 2)) ?
                             prev : /* cswap success */
                             ((T)ucs::rand() * (T)ucs::rand());
        *(T*)expected_data = atomic_op_result(op, value, prev, reply_data);

        ucp_request_param_t param;
        param.op_attr_mask = UCP_OP_ATTR_FIELD_REPLY_BUFFER;
        param.reply_buffer = &reply_data;

        ucs_status_t status = do_atomic(op, size, target_ptr, rkey, value, param);
        ASSERT_UCS_OK(status);

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
        test_all_opcodes(send_func, num_iters, op_mask, true);
        test_all_opcodes(send_func, num_iters, op_mask, false);
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
                           ucp_rkey_h rkey, T value, ucp_request_param_t &param) {

        param.op_attr_mask |= UCP_OP_ATTR_FIELD_DATATYPE;
        param.datatype      = ucp_dt_make_contig(sizeof(T));

        EXPECT_EQ(sizeof(T), size);
        ucs_status_ptr_t status_ptr = ucp_atomic_op_nbx(sender().ep(), op,
                                                        &value, 1,
                                                        (uintptr_t)target_ptr,
                                                        rkey, &param);
        return request_wait(status_ptr);
    }

    void test_all_opcodes(send_func_t send_func, unsigned num_iters,
                          uint64_t op_mask, int is_ep_flush) {
        ucs::detail::message_stream ms("INFO");

        unsigned op_value;
        ucs_for_each_bit(op_value, op_mask) {
            ucp_atomic_op_t op = static_cast<ucp_atomic_op_t>(op_value);
            ms << opcode_name(op) << " ";
            test_xfer(send_func, sizeof(T), num_iters, sizeof(T),
                      UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_HOST, 0,
                      is_ep_flush, &op);
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

UCP_INSTANTIATE_TEST_CASE(test_ucp_atomic32)

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

UCP_INSTANTIATE_TEST_CASE(test_ucp_atomic64)
