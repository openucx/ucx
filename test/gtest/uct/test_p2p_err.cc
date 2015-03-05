/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_p2p_test.h"

#include <functional>

class uct_p2p_err_test : public uct_p2p_test {
public:

    enum operation {
        OP_PUT_SHORT,
        OP_PUT_BCOPY,
        OP_PUT_ZCOPY,
        OP_AM_SHORT
    };

    static void log_handler(const char *file, unsigned line, const char *function,
                            unsigned level, const char *prefix, const char *message,
                            va_list ap)
    {
        char buf[200] = {0};
        if (level <= UCS_LOG_LEVEL_WARN) {
            va_list ap_copy;
            va_copy(ap_copy, ap); /* Create a copy of arglist, to use it 2nd time */

            ucs_log_default_handler(file, line, function, UCS_LOG_LEVEL_DEBUG, prefix, message, ap);
            vsnprintf(buf, sizeof(buf), message, ap_copy);
            va_end(ap_copy);

            UCS_TEST_MESSAGE << "   < " << buf << " >";
            ++error_count;
        } else {
            ucs_log_default_handler(file, line, function, level, prefix, message, ap);
        }
    }

    void test_error_run(enum operation op, uint8_t am_id,
                        void *buffer, size_t length, uct_mem_h memh,
                        uint64_t remote_addr, uct_rkey_t rkey)
    {
        error_count = 0;

        ucs_log_set_handler(log_handler);
        ucs_status_t status;
        do {
            switch (op) {
            case OP_PUT_SHORT:
                status = uct_ep_put_short(sender_ep(), buffer, length,
                                          remote_addr, rkey);
                break;
            case OP_PUT_BCOPY:
                status = uct_ep_put_bcopy(sender_ep(), (uct_pack_callback_t)memcpy,
                                          buffer, length, remote_addr, rkey);
                break;
            case OP_PUT_ZCOPY:
                status = uct_ep_put_zcopy(sender_ep(), buffer, length, memh,
                                          remote_addr, rkey, NULL);
                break;
            case OP_AM_SHORT:
                status = uct_ep_am_short(sender_ep(), am_id, 0, buffer, length);
                break;
            }
        } while (status == UCS_ERR_WOULD_BLOCK);

        if (status != UCS_OK && status != UCS_INPROGRESS) {
            last_error = status;
        } else {
            /* Flush async events */
            wait_for_remote();
            ucs::safe_usleep(1e4);
        }

        ucs_log_set_handler(ucs_log_default_handler);
        EXPECT_GT(error_count, 0u);
    }

    static unsigned error_count;
    static ucs_status_t last_error;

};

unsigned uct_p2p_err_test::error_count = 0;
ucs_status_t uct_p2p_err_test::last_error = UCS_OK;


UCS_TEST_P(uct_p2p_err_test, local_access_error) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF);
    mapped_buffer sendbuf(16, 1, 1, sender());
    mapped_buffer recvbuf(16, 1, 2, receiver());

    const size_t offset = 4 * 1024 * 1024;
    test_error_run(OP_PUT_ZCOPY, 0,
                   (char*)sendbuf.ptr() + offset, sendbuf.length() + offset,
                   sendbuf.memh(), recvbuf.addr(), recvbuf.rkey());

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, remote_access_error) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF);
    mapped_buffer sendbuf(16, 1, 1, sender());
    mapped_buffer recvbuf(16, 1, 2, receiver());

    const size_t offset = 4 * 1024 * 1024;
    test_error_run(OP_PUT_ZCOPY, 0,
                   (char*)sendbuf.ptr() + offset, sendbuf.length() + offset,
                   sendbuf.memh(), recvbuf.addr() + 4, recvbuf.rkey());

    recvbuf.pattern_check(2);
}

#if ENABLE_PARAMS_CHECK
UCS_TEST_P(uct_p2p_err_test, invalid_bcopy_length) {
    check_caps(UCT_IFACE_FLAG_PUT_BCOPY);
    size_t max_bcopy = sender().iface_attr().cap.put.max_bcopy;
    if (max_bcopy > (2 * 1024 * 1024)) {
        UCS_TEST_SKIP_R("max_bcopy too large");
    }

    mapped_buffer sendbuf(max_bcopy + 1, 1, 1, sender());
    mapped_buffer recvbuf(max_bcopy + 1, 1, 2, receiver());

    test_error_run(OP_PUT_BCOPY, 0, sendbuf.ptr(), sendbuf.length(),
                   UCT_INVALID_MEM_HANDLE, recvbuf.addr(), recvbuf.rkey());

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, invalid_short_length) {
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);
    size_t max_short = sender().iface_attr().cap.put.max_short;
    if (max_short > (2 * 1024 * 1024)) {
        UCS_TEST_SKIP_R("max_short too large");
    }

    mapped_buffer sendbuf(max_short + 1, 1, 1, sender());
    mapped_buffer recvbuf(max_short + 1, 1, 2, receiver());

    test_error_run(OP_PUT_SHORT, 0, sendbuf.ptr(), sendbuf.length(), UCT_INVALID_MEM_HANDLE,
                   recvbuf.addr(), recvbuf.rkey());

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, invalid_am_id) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT);

    mapped_buffer sendbuf(4, 1, 2, sender());

    test_error_run(OP_AM_SHORT, UCT_AM_ID_MAX, sendbuf.ptr(), sendbuf.length(),
                   UCT_INVALID_MEM_HANDLE, 0, UCT_INVALID_RKEY);
}
#endif

UCT_INSTANTIATE_TEST_CASE(uct_p2p_err_test)
