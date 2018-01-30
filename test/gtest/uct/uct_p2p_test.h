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
    uct_p2p_test(size_t rx_headroom, uct_error_handler_t err_handler = NULL);

    static std::vector<const resource*> enum_resources(const std::string& tl_name);

    virtual void init();
    virtual void cleanup();

    UCS_TEST_BASE_IMPL;
protected:
    typedef ucs_status_t (uct_p2p_test::* send_func_t)(uct_ep_h ep,
                                                       const mapped_buffer &,
                                                       const mapped_buffer &);

    enum uct_p2p_test_flags {
        TEST_UCT_FLAG_DIR_SEND_TO_RECV = UCS_BIT(0),
        TEST_UCT_FLAG_SEND_ZCOPY       = UCS_BIT(1),
        TEST_UCT_FLAG_RECV_ZCOPY       = UCS_BIT(2),
    };

    struct completion {
        uct_p2p_test     *self;
        uct_completion_t uct;
    };

    struct p2p_resource : public resource {
        virtual std::string name() const;
        bool loopback;
    };

    virtual void test_xfer(send_func_t send, size_t length, unsigned flags,
                           uct_memory_type_t mem_type);
    void test_xfer_multi(send_func_t send, size_t min_length, size_t max_length,
                         unsigned flags);
    void test_xfer_multi_mem_type(send_func_t send, size_t min_length, size_t max_length,
                                  unsigned flags, uct_memory_type_t mem_type);
    void blocking_send(send_func_t send, uct_ep_h ep, const mapped_buffer &sendbuf,
                       const mapped_buffer &recvbuf, bool wait_for_completion);
    void wait_for_remote();
    entity& sender();
    uct_ep_h sender_ep();
    entity& receiver();
    uct_completion_t *comp();

private:
    template <typename O>
    void test_xfer_print(O& os, send_func_t send, size_t length,
                         unsigned flags, uct_memory_type_t mem_type);

    static void completion_cb(uct_completion_t *self, ucs_status_t status);

    static ucs_log_func_rc_t
    log_handler(const char *file, unsigned line, const char *function,
                ucs_log_level_t level, const char *prefix, va_list ap);

    static int             log_data_count;
    static ucs_log_level_t orig_log_level;

    const size_t        m_rx_headroom;
    uct_error_handler_t m_err_handler;
    bool                m_null_completion;
    completion          m_completion;
    unsigned            m_completion_count;
};


#endif
