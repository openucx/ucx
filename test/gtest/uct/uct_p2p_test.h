/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_P2P_TEST_H_
#define UCT_P2P_TEST_H_

#include "uct_test.h"

/**
 * Point-to-point UCT test.
 */
class uct_p2p_test : public uct_test {
public:
    uct_p2p_test(size_t rx_headroom);

    static std::vector<const resource*> enum_resources(const std::string& tl_name);

    virtual void init();
    virtual void cleanup();

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

    struct p2p_resource : public resource {
        virtual std::string name() const;
        bool loopback;
    };

    virtual void test_xfer(send_func_t send, size_t length, direction_t direction);
    void test_xfer_multi(send_func_t send, size_t min_length, size_t max_length,
                         direction_t direction);
    void blocking_send(send_func_t send, uct_ep_h ep, const mapped_buffer &sendbuf,
                       const mapped_buffer &recvbuf);
    void wait_for_remote();
    const entity& sender() const;
    uct_ep_h sender_ep() const;
    const entity& receiver() const;
    uct_completion_t *comp();

private:
    template <typename O>
    void test_xfer_print(O& os, send_func_t send, size_t length,
                         direction_t direction);

    static void completion_cb(uct_completion_t *self);

    static ucs_log_func_rc_t
    log_handler(const char *file, unsigned line, const char *function,
                ucs_log_level_t level, const char *prefix, const char *message,
                va_list ap);

    static int             log_data_count;
    static ucs_log_level_t orig_log_level;

    const size_t m_rx_headroom;
    bool         m_null_completion;
    completion   m_completion;
    unsigned     m_completion_count;
};


#endif
