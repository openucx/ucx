/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_atomic.h"
#include "common/gtest.h"

extern "C" {
#include <ucp/core/ucp_types.h>
#include <ucp/proto/proto.h>
#include <ucp/core/ucp_ep.inl>
#include <uct/base/uct_iface.h>
#include <ucs/stats/stats.h>
}

class test_ucp_fence : public test_ucp_atomic {
public:
    typedef void (test_ucp_fence::* send_func_t)(entity *e, uint64_t *initial_buf,
                                                 uint64_t *result_buf, void *memheap_addr,
                                                 ucp_rkey_h rkey);

    static void send_cb(void *request, ucs_status_t status)
    {
    }

    template <typename T>
    void blocking_add(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                      void *memheap_addr, ucp_rkey_h rkey) {
        ucs_status_t status = ucp_atomic_post(e->ep(), UCP_ATOMIC_POST_OP_ADD,
                                              *initial_buf, sizeof(T),
                                              (uintptr_t)memheap_addr, rkey);
        ASSERT_UCS_OK(status);
    }

    template <typename T>
    void blocking_fadd(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                       void *memheap_addr, ucp_rkey_h rkey)
    {
        void *request = ucp_atomic_fetch_nb(e->ep(), UCP_ATOMIC_FETCH_OP_FADD,
                                            *initial_buf, (T*)result_buf, sizeof(T),
                                            (uintptr_t)memheap_addr, rkey, send_cb);
        wait(request);
    }

    template <typename T, typename F>
    void test(F f1, F f2) {
        test_fence(static_cast<send_func_t>(f1),
                   static_cast<send_func_t>(f2), sizeof(T));
    }

    class worker {
    public:
        worker(test_ucp_fence* test, send_func_t send1, send_func_t send2,
               entity* entity, ucp_rkey_h rkey, void *memheap_ptr,
               uint64_t initial_value, uint32_t* error):
            test(test), value(initial_value), result(0), error(error),
            running(true), m_rkey(rkey), m_memheap(memheap_ptr),
            m_send_1(send1), m_send_2(send2), m_entity(entity) {
            pthread_create(&m_thread, NULL, run, reinterpret_cast<void*>(this));
        }

        ~worker() {
            assert(!running);
        }

        static void *run(void *arg) {
            worker *self = reinterpret_cast<worker*>(arg);
            self->run();
            return NULL;
        }

        void join() {
            void *retval;
            pthread_join(m_thread, &retval);
            running = false;
        }

        test_ucp_fence* const test;
        uint64_t value, result;
        uint32_t* error;
        bool running;

    private:
        void run() {
            uint64_t zero = 0;

            for (int i = 0; i < 500 / ucs::test_time_multiplier(); i++) {
                (test->*m_send_1)(m_entity, &value, &result,
                                  m_memheap, m_rkey);

                m_entity->fence();

                (test->*m_send_2)(m_entity, &zero, &result,
                                  m_memheap, m_rkey);

                test->flush_worker(*m_entity);

                if (result != (uint64_t)(i+1))
                    (*error)++;

                result = 0; /* reset for the next loop */
            }
        }

        ucp_rkey_h m_rkey;
        void *m_memheap;
        send_func_t m_send_1, m_send_2;
        entity* m_entity;
        pthread_t m_thread;
    };

    void run_workers(send_func_t send1, send_func_t send2, entity* sender,
                     ucp_rkey_h rkey, void *memheap_ptr,
                     uint64_t initial_value, uint32_t* error) {
        ucs::ptr_vector<worker> m_workers;
        m_workers.clear();
        m_workers.push_back(new worker(this, send1, send2, sender, rkey,
                                       memheap_ptr, initial_value, error));
        m_workers.at(0).join();
        m_workers.clear();
    }

protected:
    void test_fence(send_func_t send1, send_func_t send2, size_t alignment) {
        static const size_t memheap_size = sizeof(uint64_t);
        uint32_t error = 0;

        sender().connect(&receiver(), get_ep_params());
        flush_worker(sender()); /* avoid deadlock for blocking amo */

        mapped_buffer buffer(memheap_size, receiver(), 0);

        EXPECT_LE(memheap_size, buffer.size());
        memset(buffer.ptr(), 0, memheap_size);

        run_workers(send1, send2, &sender(), buffer.rkey(sender()),
                    buffer.ptr(), 1, &error);

        EXPECT_EQ(error, (uint32_t)0);

        disconnect(sender());
        disconnect(receiver());
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }
};

class test_ucp_fence32 : public test_ucp_fence {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_fence::get_ctx_params();
        params.features |= UCP_FEATURE_AMO32;
        return params;
    }
};

UCS_TEST_P(test_ucp_fence32, atomic_add_fadd) {
    test<uint32_t>(&test_ucp_fence32::blocking_add<uint32_t>,
                   &test_ucp_fence32::blocking_fadd<uint32_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence32)

class test_ucp_fence64 : public test_ucp_fence {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_fence::get_ctx_params();
        params.features |= UCP_FEATURE_AMO64;
        return params;
    }
};

UCS_TEST_P(test_ucp_fence64, atomic_add_fadd) {
    test<uint64_t>(&test_ucp_fence64::blocking_add<uint64_t>,
                   &test_ucp_fence64::blocking_fadd<uint64_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence64)

#if ENABLE_STATS

#define UCP_INSTANTIATE_TEST_CASE_FENCE(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dcx,    "dc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rcx,    "rc_x")

class test_fence_op;
class test_fence_op {
public:
    test_fence_op(ucp_test *test) : m_test(test) {
    }

    virtual void op(void *buffer, size_t size, uintptr_t addr,
                    ucp_rkey_h rkey, bool wait) = 0;
    virtual void fence() = 0;
    virtual void sizes(std::vector<size_t> &s) = 0;

    uint64_t pending_cntr() const {
        return UCS_STATS_GET_COUNTER(m_ep->stats, UCT_EP_STAT_PENDING);
    }

    uint64_t fence_cntr() const {
        return UCS_STATS_GET_COUNTER(m_iface->stats, UCT_IFACE_STAT_FENCE_OP);
    }

    ucp_test *test() {return m_test;}

    uct_base_ep *ep() {return m_ep;}
    uct_base_iface *iface() {return m_iface;}

    virtual void init() {
        m_ep    = ucs_derived_of(test()->sender().ep()->uct_eps[lanes()[0]], uct_base_ep);
        m_iface = ucs_derived_of(ep()->super.iface, uct_base_iface);
    }

    void wait_request(ucs_status_ptr_t request, bool wait) {
        ucs_status_t status;

        ASSERT_UCS_PTR_OK(request);

        if (request == NULL) {
            return;
        }

        if (!wait) {
            ucp_request_free(request);
            return;
        }

        do {
            ucp_worker_progress(test()->sender().worker());
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(request);
    }

protected:
    virtual ucp_lane_index_t *lanes() = 0;

    static void send_callback(void *request, ucs_status_t status) {
    }

    ucp_test       *m_test;
    uct_base_ep    *m_ep;
    uct_base_iface *m_iface;
};

class test_fence_stat : public ucp_test {
public:
    virtual void init() {
        stats_activate();
        ucp_test::init();
    }

    void cleanup() {
        ucp_test::cleanup();
        stats_restore();
    }

    void test(test_fence_op *op, size_t size) {
        uint64_t fence_cntr   = op->fence_cntr();
        uint64_t pending_cntr = op->pending_cntr();
        ucs_status_t status;

        mapped_buffer buffer(size, receiver(), GetParam().variant);
        ucs::handle<ucp_rkey_h> rkey = buffer.rkey(sender());

        EXPECT_LE(size, buffer.size());
        memset(buffer.ptr(), 0, size);

        op->op(buffer.ptr(), size, (intptr_t)buffer.ptr(), rkey, true);
        /* counters should be zero */
        EXPECT_EQ(op->fence_cntr(), fence_cntr + 0);
        EXPECT_EQ(op->pending_cntr(), pending_cntr + 0);

        op->fence();
        op->op(buffer.ptr(), size, (intptr_t)buffer.ptr(), rkey, true);
        /* fence updated, pending - not */
        EXPECT_EQ(op->fence_cntr(), fence_cntr + 1);
        EXPECT_EQ(op->pending_cntr(), pending_cntr + 0);

        pending(op, buffer, size, rkey);
        /* pending updated, fence - not */
        EXPECT_EQ(op->fence_cntr(), fence_cntr + 1);
        EXPECT_EQ(op->pending_cntr(), pending_cntr + 1);

        op->fence();
        /* nothing updated */
        EXPECT_EQ(op->pending_cntr(), pending_cntr + 1);
        EXPECT_EQ(op->fence_cntr(), fence_cntr + 1);

        op->op(buffer.ptr(), size, (intptr_t)buffer.ptr(), rkey, false);
        /* pending updated, fence - not */
        EXPECT_EQ(op->fence_cntr(), fence_cntr + 1);
        EXPECT_EQ(op->pending_cntr(), pending_cntr + 2);

        status = ucp_worker_flush(sender().worker());
        ASSERT_UCS_OK(status);
        EXPECT_EQ(op->fence_cntr(), fence_cntr + 2);

        op->op(buffer.ptr(), size, (intptr_t)buffer.ptr(), rkey, true);
        /* reset pending counter - flush may re-schedule some requests */
        pending_cntr = op->pending_cntr();

        /* check that nothing updated */
        EXPECT_EQ(op->fence_cntr(), fence_cntr + 2);
        EXPECT_EQ(op->pending_cntr(), pending_cntr + 0);

        /* ok, we are done, just clean queues */
        status = ucp_worker_flush(sender().worker());
        ASSERT_UCS_OK(status);
    }

    void run(test_fence_op *op) {
        size_t i;
        std::vector<size_t> sizes;

        sender().connect(&receiver(), get_ep_params());
        flush_worker(sender()); /* avoid deadlock for blocking amo */

        op->init();

        op->sizes(sizes);

        for (i = 0; i < sizes.size(); i++) {
            test(op, sizes[i]);
        }

        disconnect(sender());
        disconnect(receiver());
    }

    void pending(test_fence_op *op, mapped_buffer &buffer, size_t size, ucp_rkey_h rkey) {
        uint64_t counter = op->pending_cntr();

        while (counter == op->pending_cntr()) {
            op->op(buffer.ptr(), size, (intptr_t)buffer.ptr(), rkey, false);
        }
    }
protected:
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};

class test_fence_rma : public test_fence_stat {
public:
    test_fence_rma() {
        m_env.push_back(new ucs::scoped_setenv("UCX_DC_MLX5_FENCE", "strong"));
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_MLX5_FENCE", "strong"));
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }
};

class test_fence_amo32 : public test_fence_stat {
public:
    test_fence_amo32() {
        m_env.push_back(new ucs::scoped_setenv("UCX_DC_MLX5_FENCE", "weak"));
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_MLX5_FENCE", "weak"));
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_AMO32;
        return params;
    }
};

class test_fence_amo64 : public test_fence_stat {
public:
    test_fence_amo64() {
        m_env.push_back(new ucs::scoped_setenv("UCX_DC_MLX5_FENCE", "weak"));
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_MLX5_FENCE", "weak"));
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_AMO64;
        return params;
    }
};

class test_fence_worker : public test_fence_op {
public:
    test_fence_worker(ucp_test *test) : test_fence_op(test) {
    }

    virtual void fence() {
        ucs_status_t status;

        status = ucp_worker_fence(test()->sender().worker());
        ASSERT_UCS_OK(status);
    }
};

class test_fence_put : public test_fence_worker {
public:
    test_fence_put(ucp_test *test) : test_fence_worker(test) {
    }

    void sizes(std::vector<size_t> &s) {
        uct_iface_attr_t attr;
        ucs_status_t status;

        status = uct_iface_query(&iface()->super, &attr);
        ASSERT_UCS_OK(status);

        if (attr.cap.flags & UCT_IFACE_FLAG_PUT_SHORT) {
            s.push_back(attr.cap.put.max_short / 2);
        }

        if (attr.cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
            s.push_back(attr.cap.put.max_bcopy / 2);
        }

        if (attr.cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY) {
            if (attr.cap.put.min_zcopy) {
                s.push_back(attr.cap.put.min_zcopy);
                s.push_back(attr.cap.put.min_zcopy * 2);
                s.push_back(attr.cap.put.min_zcopy * 4);
                s.push_back(attr.cap.put.min_zcopy * 8);
                s.push_back(attr.cap.put.min_zcopy * 16);
            } else if (attr.cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
                s.push_back(attr.cap.put.max_bcopy + 1);
            } else if (attr.cap.flags & UCT_IFACE_FLAG_PUT_SHORT) {
                s.push_back(attr.cap.put.max_short + 1);
            }
        }
    }

    virtual void op(void *buffer, size_t size, uintptr_t addr,
                    ucp_rkey_h rkey, bool wait) {
        ucs_status_ptr_t request;

        request = ucp_put_nb(test()->sender().ep(), buffer, size,
                             addr, rkey, send_callback);
        wait_request(request, wait);
    }

protected:
    virtual ucp_lane_index_t *lanes() {
        return ucp_ep_config(test()->sender().ep())->key.rma_lanes;
    }
};

class test_fence_get : public test_fence_put {
public:
    test_fence_get(ucp_test *test) : test_fence_put(test) {
    }

    virtual void op(void *buffer, size_t size, uintptr_t addr,
                    ucp_rkey_h rkey, bool wait) {
        ucs_status_ptr_t request;

        request = ucp_get_nb(test()->sender().ep(), buffer, size,
                             addr, rkey, send_callback);
        wait_request(request, wait);
    }
};

template <ucp_atomic_fetch_op_t OP, size_t S>
class test_fence_famo : public test_fence_worker {
public:
    test_fence_famo(ucp_test *test) : test_fence_worker(test) {
    }

    void sizes(std::vector<size_t> &s) {
        s.push_back(S);
    }

    virtual void op(void *buffer, size_t size, uintptr_t addr,
                    ucp_rkey_h rkey, bool wait) {
        ucs_status_ptr_t request;

        ASSERT_EQ(S, size);
        request = ucp_atomic_fetch_nb(test()->sender().ep(), OP, 0, buffer, S,
                                      addr, rkey, send_callback);
        wait_request(request, wait);
    }

protected:
    virtual ucp_lane_index_t *lanes() {
        return ucp_ep_config(test()->sender().ep())->key.amo_lanes;
    }
};

UCS_TEST_P(test_fence_rma, put) {
    test_fence_put op(this);
    run(&op);
}

UCS_TEST_P(test_fence_rma, get) {
    test_fence_get op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo64, fadd) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FADD, 8> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo64, swap) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_SWAP, 8> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo64, cswap) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_CSWAP, 8> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo64, fand) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FAND, 8> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo64, _for) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FOR, 8> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo64, fxor) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FXOR, 8> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo32, fadd) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FADD, 4> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo32, swap) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_SWAP, 4> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo32, cswap) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_CSWAP, 4> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo32, fand) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FAND, 4> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo32, _for) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FOR, 4> op(this);
    run(&op);
}

UCS_TEST_P(test_fence_amo32, fxor) {
    test_fence_famo<UCP_ATOMIC_FETCH_OP_FXOR, 4> op(this);
    run(&op);
}

UCP_INSTANTIATE_TEST_CASE_FENCE(test_fence_rma)
UCP_INSTANTIATE_TEST_CASE_FENCE(test_fence_amo64)
UCP_INSTANTIATE_TEST_CASE_FENCE(test_fence_amo32)

#endif /* ENABLE_STATS */
