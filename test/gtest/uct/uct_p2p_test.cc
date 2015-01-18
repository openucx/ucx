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
    for (unsigned i =0; i < 2; ++i) {
        m_entities.push_back(new entity(GetParam()));
    }

    m_entities[0].connect(m_entities[1]);
    m_entities[1].connect(m_entities[0]);

    /* Allocate completion handle and set the callback */
    m_completion = (completion*)malloc(sizeof(completion) +
                          m_entities[1].iface_attr().completion_priv_len);
    m_completion->self           = this;
    m_completion->uct.super.func = completion_cb;

    m_completion_count = 0;
}

void uct_p2p_test::cleanup() {
    free(m_completion);
    m_entities.clear();
}

void uct_p2p_test::progress() {
    for (ucs::ptr_vector<entity>::const_iterator iter = m_entities.begin();
         iter != m_entities.end();
         ++iter)
    {
        (*iter)->progress();
    }
}

void uct_p2p_test::short_progress_loop() {
    ucs_time_t end_time = ucs_get_time() + ucs_time_from_msec(1.0);
    while (ucs_get_time() < end_time) {
        progress();
    }
}

void uct_p2p_test::test_xfer(send_func_t send, size_t length) {
    UCS_TEST_SKIP;
}

template <typename O>
void uct_p2p_test::test_xfer_print(const O& os, send_func_t send, size_t length)
{
    os << std::fixed << std::setprecision(1);
    if (length < 1024) {
        os << length;
    } else if (length < 1024 * 1024) {
        os << (length / 1024.0) << "k";
    } else if (length < 1024 * 1024 * 1024) {
        os << (length / 1024.0 / 1024.0) << "m";
    } else {
        os << (length / 1024.0 / 1024.0 / 1024.0) << "g";
    }

    os << " " << std::flush;
    test_xfer(send, length);
}

void uct_p2p_test::test_xfer_multi(send_func_t send, ssize_t min_length, ssize_t max_length) {

    if (!(max_length > min_length)) {
        UCS_TEST_SKIP;
    }

    ucs::detail::message_stream ms("INFO");

    /* Run with min and max values */
    test_xfer_print(ms, send, min_length);
    test_xfer_print(ms, send, max_length);

    /*
     * Generate SQRT( log(max/min) ) random sizes
     */
    double log_min = log2(min_length + 1);
    double log_max = log2(max_length - 1);
    int count = (int)sqrt(log_max - log_min);
    for (int i = 0; i < count; ++i) {
        double exp = (rand() * (log_max - log_min)) / RAND_MAX + log_min;
        ssize_t length = (ssize_t)pow(2.0, exp);
        ucs_assert(length > min_length && length < max_length);
        test_xfer_print(ms, send, length);
    }
}

void uct_p2p_test::wait_for_local(ucs_status_t status, unsigned prev_comp_count) {
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
    get_entity(0).flush();
}

const uct_test::entity& uct_p2p_test::get_entity(unsigned index) const {
    return m_entities[index];
}

void uct_p2p_test::completion_cb(ucs_callback_t *self) {
    completion *comp = ucs_container_of(self, completion, uct);
    ++comp->self->m_completion_count;
}
