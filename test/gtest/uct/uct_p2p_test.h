/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_P2P_TEST_H_
#define UCT_P2P_TEST_H_

#include "uct_test.h"

/**
 * Point-to-point UCT test.
 */
class uct_p2p_test : public uct_test {
public:
    virtual void init();
    virtual void cleanup();

    void short_progress_loop();

    UCS_TEST_BASE_IMPL;
protected:
    typedef ucs_status_t (uct_p2p_test::* send_func_t)(uct_ep_h ep,
                                                       const mapped_buffer &,
                                                       const mapped_buffer &);

    typedef enum {
        DIRECTION_SEND_TO_RECV,
        DIRECTION_RECV_TO_SEND
    } direction_t;

    struct completion {
        uct_p2p_test     *self;
        uct_completion_t uct;
    };

    virtual void test_xfer(send_func_t send, size_t length, direction_t direction);
    void test_xfer_multi(send_func_t send, ssize_t min_length, ssize_t max_length,
                         direction_t direction);
    void wait_for_local(ucs_status_t status, unsigned prev_comp_count);
    void wait_for_remote();
    const entity& sender() const;
    uct_ep_h sender_ep() const;
    const entity& receiver() const;

    completion *m_completion;
    unsigned m_completion_count;

private:

    template <typename O>
    void test_xfer_print(const O& os, send_func_t send, size_t length,
                         direction_t direction);

    static void completion_cb(ucs_callback_t *self);
};


#endif
