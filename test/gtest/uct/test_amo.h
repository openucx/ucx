/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_TEST_AMO_H
#define UCT_TEST_AMO_H

#include "uct_test.h"

#include <vector>

class uct_amo_test : public uct_test {
public:
    class worker;
    struct completion;
    typedef ucs_status_t (uct_amo_test::* send_func_t)(uct_ep_h ep, worker& worker,
                                                       const mapped_buffer& recvbuf,
                                                       uint64_t *result, completion *comp);

    static inline unsigned num_senders() {
        return (RUNNING_ON_VALGRIND) ? 2 : 4;
    }
    static inline unsigned count() {
        return 1000 / ucs::test_time_multiplier();
    }

    uct_amo_test();
    virtual void init();
    virtual void cleanup();

    const entity& receiver();
    const entity& sender(unsigned index);
    void validate_replies(const std::vector<uint64_t>& exp_replies);
    void wait_for_remote();
    void add_reply_safe(uint64_t data);

    static uint64_t rand64();
    static uint64_t hash64(uint64_t value);

    static void atomic_reply_cb(uct_completion_t *self, ucs_status_t status);

    void run_workers(send_func_t send, const mapped_buffer& recvbuf,
                     std::vector<uint64_t> initial_values, bool advance);

    struct completion {
        union {
            uct_amo_test *self;
            worker       *w;
        };
        uint64_t         result;
        uct_completion_t uct;
    };

    class worker {
    public:
        worker(uct_amo_test* test, send_func_t send, const mapped_buffer& recvbuf,
               const entity& entity, uint64_t initial_value, bool advance);
        ~worker();

        completion *get_completion(unsigned index);
        static void* run(void *arg);
        void join();

        uct_amo_test* const test;
        uint64_t            value;
        unsigned            count;
        bool                running;

    private:
        void run();

        send_func_t          m_send;
        const bool           m_advance;
        const mapped_buffer& m_recvbuf;
        const entity&        m_entity;
        pthread_t            m_thread;
        std::vector<completion> m_completions;
    };


protected:

    ucs::ptr_vector<worker> m_workers;
    pthread_spinlock_t      m_replies_lock;
    std::vector<uint64_t>   m_replies;
};


#endif
