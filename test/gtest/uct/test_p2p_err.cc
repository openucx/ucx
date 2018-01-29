/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"

#include <functional>

class uct_p2p_err_test : public uct_p2p_test {
public:

    enum operation {
        OP_PUT_SHORT,
        OP_PUT_BCOPY,
        OP_PUT_ZCOPY,
        OP_AM_SHORT,
        OP_AM_BCOPY,
        OP_AM_ZCOPY
    };

    struct pack_arg {
        void   *buffer;
        size_t length;
    };

    uct_p2p_err_test() :
        uct_p2p_test(0, uct_error_handler_t(ucs_empty_function_return_success)) {
    }

    static size_t pack_cb(void *dest, void *arg)
    {
        pack_arg *pa = (pack_arg*)arg;
        memcpy(dest, pa->buffer, pa->length);
        return pa->length;
    }

    void test_error_run(enum operation op, uint8_t am_id,
                        void *buffer, size_t length, uct_mem_h memh,
                        uint64_t remote_addr, uct_rkey_t rkey,
                        const std::string& error_pattern)
    {
        pack_arg arg;

        wrap_errors();

        UCS_TEST_SCOPE_EXIT() { restore_errors(); } UCS_TEST_SCOPE_EXIT_END

        ucs_status_t status = UCS_OK;
        ssize_t packed_len;
        do {
            switch (op) {
            case OP_PUT_SHORT:
                status = uct_ep_put_short(sender_ep(), buffer, length,
                                          remote_addr, rkey);
                break;
            case OP_PUT_BCOPY:
                arg.buffer = buffer;
                arg.length = length;
                packed_len = uct_ep_put_bcopy(sender_ep(), pack_cb, &arg, remote_addr,
                                              rkey);
                status = (packed_len >= 0) ? UCS_OK : (ucs_status_t)packed_len;
                break;
            case OP_PUT_ZCOPY:
            {
                UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buffer, length, memh,
                                        sender().iface_attr().cap.put.max_iov);
                status = uct_ep_put_zcopy(sender_ep(), iov, iovcnt,
                                          remote_addr, rkey, NULL);
            }
                break;
            case OP_AM_SHORT:
                status = uct_ep_am_short(sender_ep(), am_id, 0, buffer, length);
                break;
            case OP_AM_BCOPY:
                arg.buffer = buffer;
                arg.length = length;
                packed_len = uct_ep_am_bcopy(sender_ep(), am_id, pack_cb, &arg, 0);
                status = (packed_len >= 0) ? UCS_OK : (ucs_status_t)packed_len;
                break;
            case OP_AM_ZCOPY:
            {
                UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buffer, 1, memh, 1);
                status = uct_ep_am_zcopy(sender_ep(), am_id, buffer, length,
                                         iov, iovcnt, 0, NULL);
            }
                break;
            }
        } while (status == UCS_ERR_NO_RESOURCE);

        if (status != UCS_OK && status != UCS_INPROGRESS) {
            last_error = status;
        } else {
            /* Flush async events */
            wait_for_remote();
            ucs::safe_usleep(1e4);
        }

        /* Count how many error messages match/don't match the given pattern */
        size_t num_matched   = 0;
        size_t num_unmatched = 0;
        for (std::vector<std::string>::iterator iter = m_errors.begin();
                        iter != m_errors.end(); ++iter) {
            if (iter->find(error_pattern) != iter->npos) {
                ++num_matched;
            } else {
                ++num_unmatched;
            }
        }

        EXPECT_GT(num_matched, 0ul) <<
                        "No error which contains the string '" << error_pattern <<
                        "' has occurred during the test";
        EXPECT_EQ(0ul, num_unmatched) <<
                        "Unexpected error(s) occurred during the test";
    }

    static void* get_unused_address(size_t length)
    {
        void *address = NULL;
        ucs_status_t status = ucs_mmap_alloc(&length, &address, 0
                                             UCS_MEMTRACK_NAME("test_dummy"));
        ASSERT_UCS_OK(status, << "length = " << length);
        status = ucs_mmap_free(address, length);
        ASSERT_UCS_OK(status);
        /* coverity[use_after_free] */
        return address;
    }

    static ucs_status_t last_error;

};

ucs_status_t uct_p2p_err_test::last_error = UCS_OK;


UCS_TEST_P(uct_p2p_err_test, local_access_error) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF);
    mapped_buffer sendbuf(16, 1, sender());
    mapped_buffer recvbuf(16, 2, receiver());

    test_error_run(OP_PUT_ZCOPY, 0,
                   get_unused_address(sendbuf.length()), sendbuf.length(),
                   sendbuf.memh(), recvbuf.addr(), recvbuf.rkey(),
                   "");

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, remote_access_error) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM);
    mapped_buffer sendbuf(16, 1, sender());
    mapped_buffer recvbuf(16, 2, receiver());

    test_error_run(OP_PUT_ZCOPY, 0,
                   sendbuf.ptr(), sendbuf.length(), sendbuf.memh(),
                   (uintptr_t)get_unused_address(recvbuf.length()), recvbuf.rkey(),
                   "");

    recvbuf.pattern_check(2);
}

#if ENABLE_PARAMS_CHECK
UCS_TEST_P(uct_p2p_err_test, invalid_put_short_length) {
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);
    size_t max_short = sender().iface_attr().cap.put.max_short;
    if (max_short > (2 * 1024 * 1024)) {
        UCS_TEST_SKIP_R("max_short too large");
    }

    mapped_buffer sendbuf(max_short + 1, 1, sender());
    mapped_buffer recvbuf(max_short + 1, 2, receiver());

    test_error_run(OP_PUT_SHORT, 0, sendbuf.ptr(), sendbuf.length(),
                   UCT_MEM_HANDLE_NULL, recvbuf.addr(), recvbuf.rkey(),
                   "length");

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, invalid_put_bcopy_length) {
    check_caps(UCT_IFACE_FLAG_PUT_BCOPY | UCT_IFACE_FLAG_ERRHANDLE_BCOPY_LEN);
    size_t max_bcopy = sender().iface_attr().cap.put.max_bcopy;
    if (max_bcopy > (2 * 1024 * 1024)) {
        UCS_TEST_SKIP_R("max_bcopy too large");
    }

    mapped_buffer sendbuf(max_bcopy + 1, 1, sender());
    mapped_buffer recvbuf(max_bcopy + 1, 2, receiver());

    test_error_run(OP_PUT_BCOPY, 0, sendbuf.ptr(), sendbuf.length(),
                   UCT_MEM_HANDLE_NULL, recvbuf.addr(), recvbuf.rkey(),
                   "length");

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, invalid_am_short_length) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT);
    size_t max_short = sender().iface_attr().cap.am.max_short;
    if (max_short > (2 * 1024 * 1024)) {
        UCS_TEST_SKIP_R("max_short too large");
    }

    mapped_buffer sendbuf(max_short + 1 - sizeof(uint64_t), 1, sender());
    mapped_buffer recvbuf(max_short + 1,                    2, receiver());

    test_error_run(OP_AM_SHORT, 0, sendbuf.ptr(), sendbuf.length(),
                   UCT_MEM_HANDLE_NULL, recvbuf.addr(), recvbuf.rkey(),
                   "length");

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, invalid_am_bcopy_length) {
    check_caps(UCT_IFACE_FLAG_AM_BCOPY | UCT_IFACE_FLAG_ERRHANDLE_BCOPY_LEN);
    size_t max_bcopy = sender().iface_attr().cap.am.max_bcopy;
    if (max_bcopy > (2 * 1024 * 1024)) {
        UCS_TEST_SKIP_R("max_bcopy too large");
    }

    mapped_buffer sendbuf(max_bcopy + 1, 1, sender());
    mapped_buffer recvbuf(max_bcopy + 1, 2, receiver());

    test_error_run(OP_AM_BCOPY, 0, sendbuf.ptr(), sendbuf.length(),
                   UCT_MEM_HANDLE_NULL, recvbuf.addr(), recvbuf.rkey(),
                   "length");

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, invalid_am_zcopy_hdr_length) {
    check_caps(UCT_IFACE_FLAG_AM_ZCOPY);
    size_t max_hdr = sender().iface_attr().cap.am.max_hdr;
    if (max_hdr > (2 * 1024 * 1024)) {
        UCS_TEST_SKIP_R("max_hdr too large");
    }
    if (max_hdr + 2 > sender().iface_attr().cap.am.max_bcopy) {
        UCS_TEST_SKIP_R("max_hdr + 2 exceeds maximal bcopy size");
    }

    /* Send header of (max_hdr+1) and payload length 1 */
    mapped_buffer sendbuf(max_hdr + 1, 1, sender());
    mapped_buffer recvbuf(max_hdr + 2, 2, receiver());

    test_error_run(OP_AM_ZCOPY, 0, sendbuf.ptr(), sendbuf.length(),
                   sendbuf.memh(), recvbuf.addr(), recvbuf.rkey(),
                   "length");

    recvbuf.pattern_check(2);
}

UCS_TEST_P(uct_p2p_err_test, invalid_am_id) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT);

    mapped_buffer sendbuf(4, 2, sender());

    test_error_run(OP_AM_SHORT, UCT_AM_ID_MAX, sendbuf.ptr(), sendbuf.length(),
                   UCT_MEM_HANDLE_NULL, 0, UCT_INVALID_RKEY,
                   "active message id");
}
#endif

UCT_INSTANTIATE_TEST_CASE(uct_p2p_err_test)
