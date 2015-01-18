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

    void progress();
    void short_progress_loop();

    UCS_TEST_BASE_IMPL;
protected:
    typedef ucs_status_t (uct_p2p_test::* send_func_t)(const entity&,
                                                       const mapped_buffer &,
                                                       const mapped_buffer &);

    struct completion {
        uct_p2p_test     *self;
        uct_completion_t uct;
    };

    const entity &get_entity(unsigned index) const;

    virtual void test_xfer(send_func_t send, size_t length);
    void test_xfer_multi(send_func_t send, ssize_t min_length, ssize_t max_length);
    void wait_for_local(ucs_status_t status, unsigned prev_comp_count);
    void wait_for_remote();

    completion *m_completion;
    unsigned m_completion_count;

private:
    ucs::ptr_vector<entity> m_entities;

    template <typename O>
    void test_xfer_print(const O& os, send_func_t send, size_t length);

    static void completion_cb(ucs_callback_t *self);
};


#endif
