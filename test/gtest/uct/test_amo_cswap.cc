/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_cswap_test : public uct_amo_test {
public:

    static const uint64_t MISS = 0;

    template <typename T>
    static void cswap_reply_cb(uct_completion_t *self, ucs_status_t status) {
        completion *comp = ucs_container_of(self, completion, uct);
        worker* w = comp->w;
        T dataval = comp->result;

        /* Compare after casting to T, since w->value is always 64 bit */
        if (dataval == (T)w->value) {
            w->test->add_reply_safe(dataval); /* Swapped */
        } else {
            w->test->add_reply_safe(MISS); /* Miss value */
        }

        w->value = (T)hash64(w->value); /* Move to next value */
        --w->count; /* Allow one more operation */
    }

    ucs_status_t cswap32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                         uint64_t *result, completion *comp) {
        if (worker.count > 0) {
            return UCS_ERR_NO_RESOURCE; /* Don't proceed until got a reply */
        }
        comp->uct.func = cswap_reply_cb<uint32_t>;
        comp->w        = &worker;
        // TODO will not work if completes immediately
        return uct_ep_atomic_cswap32(ep, worker.value, hash64(worker.value),
                                     recvbuf.addr(), recvbuf.rkey(),
                                     (uint32_t*)result, &comp->uct);
    }

    ucs_status_t cswap64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                         uint64_t *result, completion *comp) {
        if (worker.count > 0) {
            return UCS_ERR_NO_RESOURCE; /* Don't proceed until got a reply */
        }
        comp->uct.func = cswap_reply_cb<uint64_t>;
        comp->w        = &worker;
        // TODO will not work if completes immediately
        return uct_ep_atomic_cswap64(ep, worker.value, hash64(worker.value),
                                     recvbuf.addr(), recvbuf.rkey(),
                                     result, &comp->uct);
    }

    template <typename T>
    void test_cswap(send_func_t send) {
        /*
         * Method: All workers try to create a swap chain using the same series of
         * values. But only one worker should be able to advance to the next
         * value every time.
         * This test is different because it sends the next request only after
         * getting a reply.
         */

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        /* Set ransom initial value */
        T initial_value = rand64();
        *(T*)recvbuf.ptr() = initial_value;

        T value = initial_value;
        std::vector<uint64_t> exp_replies;
        for (unsigned i = 0; i < count(); ++i) {
            exp_replies.push_back(value);
            value = hash64(value);
        }

        /* Expect N-1 cswap misses for each value */
        for (unsigned i = 0; i < count() * (num_senders() - 1); ++i) {
            exp_replies.push_back(static_cast<T>(MISS));
        }

        run_workers(send, recvbuf, std::vector<uint64_t>(num_senders(), initial_value),
                    false);

        validate_replies(exp_replies);

        wait_for_remote();
        EXPECT_EQ(value, *(T*)recvbuf.ptr());
    }
};


UCS_TEST_P(uct_amo_cswap_test, cswap32) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_CSWAP32);
    test_cswap<uint32_t>(static_cast<send_func_t>(&uct_amo_cswap_test::cswap32));
}

UCS_TEST_P(uct_amo_cswap_test, cswap64) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_CSWAP64);
    test_cswap<uint64_t>(static_cast<send_func_t>(&uct_amo_cswap_test::cswap64));
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_cswap_test)
