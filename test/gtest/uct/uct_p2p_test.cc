/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_p2p_test.h"
extern "C" {
#include <ucs/time/time.h>
}

void uct_p2p_test::init() {
    uct_test::init();

    /* Create 2 connected endpoints */
    entity *e1 = new entity(GetParam());
    entity *e2 = new entity(GetParam());
    e1->add_ep();
    e2->add_ep();
    e1->connect(0, *e2, 0);
    e2->connect(0, *e1, 0);

    m_entities.push_back(e1);
    m_entities.push_back(e2);

    /* Allocate completion handle and set the callback */
    m_completion = (completion*)malloc(sizeof(completion) +
                                       sender().iface_attr().completion_priv_len);
    m_completion->self           = this;
    m_completion->uct.super.func = completion_cb;

    m_completion_count = 0;
}

void uct_p2p_test::cleanup() {
    free(m_completion);
    uct_test::cleanup();
}

void uct_p2p_test::short_progress_loop() {
    ucs_time_t end_time = ucs_get_time() + ucs_time_from_msec(1.0);
    while (ucs_get_time() < end_time) {
        progress();
    }
}

void uct_p2p_test::test_xfer(send_func_t send, size_t length, direction_t direction) {
    UCS_TEST_SKIP;
}

class size {
public:
    explicit size(size_t value) : m_value(value) {}

    size_t value() const {
        return m_value;
    }
private:
    size_t m_value;
};

template <typename O>
static O& operator<<(O& os, const size& sz)
{
    size_t v = sz.value();

    os << std::fixed << std::setprecision(1);
    if (v < 1024) {
        os << v;
    } else if (v < 1024 * 1024) {
        os << (v / 1024.0) << "k";
    } else if (v < 1024 * 1024 * 1024) {
        os << (v / 1024.0 / 1024.0) << "m";
    } else {
        os << (v / 1024.0 / 1024.0 / 1024.0) << "g";
    }

    return os;
}

template <typename O>
void uct_p2p_test::test_xfer_print(const O& os, send_func_t send, size_t length,
                                   direction_t direction)
{
    os << size(length) << " " << std::flush;
    test_xfer(send, length, direction);
}

void uct_p2p_test::test_xfer_multi(send_func_t send, ssize_t min_length,
                                   ssize_t max_length, direction_t direction) {

    if (max_length > 1 * 1024 * 1024) {
        max_length /= ucs::test_time_multiplier();
    }

    if (max_length <= min_length) {
        UCS_TEST_SKIP;
    }

    ucs::detail::message_stream ms("INFO");

    /* Run with min and max values */
    test_xfer_print(ms, send, min_length, direction);
    test_xfer_print(ms, send, max_length, direction);

    /*
     * Generate SQRT( log(max/min) ) random sizes
     */
    double log_min = log2(min_length + 1);
    double log_max = log2(max_length - 1);

    /* How many times to repeat */
    int repeat_count;
    repeat_count = (1 * 1024 * 1024) / ((max_length + min_length) / 2);
    if (repeat_count > 3000) {
        repeat_count = 3000;
    }
    repeat_count /= ucs::test_time_multiplier();
    if (repeat_count == 0) {
        repeat_count = 1;
    }

    ms << repeat_count << "x{" << size(min_length) << ".." << size(max_length) << "} "
                    << std::flush;
    for (int i = 0; i < repeat_count; ++i) {
        double exp = (rand() * (log_max - log_min)) / RAND_MAX + log_min;
        ssize_t length = (ssize_t)pow(2.0, exp);
        ucs_assert(length > min_length && length < max_length);
        test_xfer(send, length, direction);
    }

    sender().flush();
}

void uct_p2p_test::blocking_send(send_func_t send, uct_ep_h ep,
                                 const mapped_buffer &sendbuf,
                                 const mapped_buffer &recvbuf,
                                 unsigned prev_comp_count)
{
    ucs_status_t status;
    do {
        status = (this->*send)(ep, sendbuf, recvbuf);
    } while (status == UCS_ERR_WOULD_BLOCK);
    if (status == UCS_OK) {
        return;
    } else if (status == UCS_INPROGRESS) {
        while (m_completion_count <= prev_comp_count) {
            progress();
        }
    } else {
        UCS_TEST_ABORT(ucs_status_string(status));
    }
}

void uct_p2p_test::wait_for_remote() {
    sender().flush();
}

const uct_test::entity& uct_p2p_test::sender() const {
    return ent(0);
}

uct_ep_h uct_p2p_test::sender_ep() const {
    return sender().ep(0);
}

const uct_test::entity& uct_p2p_test::receiver() const {
    return ent(1);
}

void uct_p2p_test::completion_cb(ucs_callback_t *self) {
    completion *comp = ucs_container_of(self, completion, uct);
    ++comp->self->m_completion_count;
}
