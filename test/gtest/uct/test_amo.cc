/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "test_amo.h"

#include <functional>


uct_amo_test::uct_amo_test() {
    pthread_spin_init(&m_replies_lock, 0);
}

void uct_amo_test::init() {
    uct_test::init();

    srand48(rand());

    entity *receiver = new entity(GetParam());
    m_entities.push_back(receiver);

    for (unsigned i = 0; i < num_senders(); ++i) {
        receiver->add_ep();
        entity *sender = new entity(GetParam());
        m_entities.push_back(sender);
        sender->add_ep();
        sender->connect(0, *receiver, i);
        receiver->connect(i, *sender, 0);
    }
}

uint64_t uct_amo_test::rand64() {
    return (mrand48() << 32) | (uint32_t)mrand48();
}

uint64_t uct_amo_test::hash64(uint64_t value) {
    return value * 171711717;
}

void uct_amo_test::atomic_reply_cb(void *arg, uint64_t data) {
    uct_amo_test *self = reinterpret_cast<uct_amo_test*>(arg);
    self->add_reply_safe(data);
}

void uct_amo_test::add_reply_safe(uint64_t data) {
    pthread_spin_lock(&m_replies_lock);
    m_replies.push_back(data);
    pthread_spin_unlock(&m_replies_lock);
}

const uct_amo_test::entity& uct_amo_test::receiver() {
    return m_entities.at(0);
}

const uct_amo_test::entity& uct_amo_test::sender(unsigned index) {
    return m_entities.at(1 + index);
}

void uct_amo_test::validate_replies(const std::vector<uint64_t>& exp_replies) {

    /* Count histogram of expected replies */
    std::map<uint64_t, int> exp_h;
    for (std::vector<uint64_t>::const_iterator iter = exp_replies.begin();
                    iter != exp_replies.end(); ++iter) {
        ++exp_h[*iter];
    }

    for (ucs::ptr_vector<worker>::const_iterator iter = m_workers.begin();
                    iter != m_workers.end(); ++iter)
    {
        ucs_assert(!(*iter)->running);
    }

    /* Workers should not be running now.
     * Count a histogram of actual replies.
     */
    unsigned count = 0;
    std::map<uint64_t, int> h;

    while (count < exp_replies.size()) {
        while (m_replies.empty()) {
            progress();
        }

        ++h[m_replies.back()];
        m_replies.pop_back();
        ++count;
    }

    /* Destroy workers only after getting all replies, because reply callback
     * may use the worker object (e.g CSWAP test). */
    m_workers.clear();

    /* Every reply should be present exactly once */
    for (std::map<uint64_t, int>::const_iterator iter = exp_h.begin();
                    iter != exp_h.end(); ++iter)
    {
        if (h[iter->first] != iter->second) {
            UCS_TEST_ABORT("Reply " << iter->first << " appeared " << h[iter->first] <<
                           " times; expected: " << iter->second);
        }
        h.erase(iter->first);
    }

    if (!h.empty()) {
        UCS_TEST_ABORT("Got some unexpected replies, e.g: " << h.begin()->first <<
                       " (" << h.begin()->second << " times)");
    }
}

void uct_amo_test::wait_for_remote() {
    for (unsigned i = 0; i < num_senders(); ++i) {
        sender(i).flush();
    }
}

void uct_amo_test::run_workers(send_func_t send, const mapped_buffer& recvbuf,
                               std::vector<uint64_t> initial_values, bool advance)
{
    m_workers.clear();

    for (unsigned i = 0; i < num_senders(); ++i) {
        m_workers.push_back(new worker(this, send, recvbuf, sender(i),
                                       initial_values[i], advance));
    }

    for (unsigned i = 0; i < num_senders(); ++i) {
        m_workers.at(i).join();
    }
}

uct_amo_test::worker::worker(uct_amo_test* test, send_func_t send,
                             const mapped_buffer& recvbuf, const entity& entity,
                             uint64_t initial_value, bool advance) :
    test(test), value(initial_value), count(0), running(true),
    m_send(send), m_advance(advance), m_recvbuf(recvbuf), m_entity(entity)

{
    pthread_create(&m_thread, NULL, run, reinterpret_cast<void*>(this));

}

uct_amo_test::worker::~worker() {
    ucs_assert(!running);
}

void* uct_amo_test::worker::run(void *arg) {
    worker *self = reinterpret_cast<worker*>(arg);
    self->run();
    return NULL;
}

void uct_amo_test::worker::run() {
    for (unsigned i = 0; i < uct_amo_test::count(); ++i) {
        ucs_status_t status;
        status = (test->*m_send)(m_entity.ep(0), *this, m_recvbuf);
        while (status == UCS_ERR_WOULD_BLOCK) {
            m_entity.progress();
            status = (test->*m_send)(m_entity.ep(0), *this, m_recvbuf);
        }
        if ((status != UCS_OK) && (status != UCS_INPROGRESS)) {
            UCS_TEST_ABORT(ucs_status_string(status));
        }
        ++count;
        if (m_advance) {
            value = hash64(value);
        }
    }
}

void uct_amo_test::worker::join() {
    void *retval;
    pthread_join(m_thread, &retval);
    running = false;
}

