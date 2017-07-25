/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"

extern "C" {
#include <ucs/arch/atomic.h>
}
#include <functional>

class uct_p2p_mix_test : public uct_p2p_test {
public:

    typedef ucs_status_t
            (uct_p2p_mix_test::* send_func_t)(const mapped_buffer &sendbuf,
                                              const mapped_buffer &recvbuf,
                                              uct_completion_t *comp);

    static const uint8_t AM_ID    = 1;
    static const size_t  MAX_SIZE = 256;

    uct_p2p_mix_test() : uct_p2p_test(0), m_send_size(0) {
    }

    static ucs_status_t am_callback(void *arg, void *data, size_t length,
                                    unsigned flags)
    {
        ucs_atomic_add32(&am_pending, -1);
        return UCS_OK;
    }

    static void completion_callback(uct_completion_t *comp, ucs_status_t status)
    {
        ASSERT_UCS_OK(status);
    }

    ucs_status_t swap64(const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf,
                        uct_completion_t *comp)
    {
        return uct_ep_atomic_swap64(sender().ep(0), 1, recvbuf.addr(),
                                    recvbuf.rkey(), (uint64_t*)sendbuf.ptr(),
                                    comp);
    }

    ucs_status_t cswap64(const mapped_buffer &sendbuf,
                         const mapped_buffer &recvbuf,
                         uct_completion_t *comp)
    {
        return uct_ep_atomic_cswap64(sender().ep(0), 0, 1, recvbuf.addr(),
                                     recvbuf.rkey(), (uint64_t*)sendbuf.ptr(),
                                     comp);
    }

    ucs_status_t fadd32(const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf,
                        uct_completion_t *comp)
    {
        return uct_ep_atomic_fadd32(sender().ep(0), 1, recvbuf.addr(),
                                    recvbuf.rkey(), (uint32_t*)sendbuf.ptr(),
                                    comp);
    }

    ucs_status_t swap32(const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf,
                        uct_completion_t *comp)
    {
        return uct_ep_atomic_swap32(sender().ep(0), 1, recvbuf.addr(),
                                    recvbuf.rkey(), (uint32_t*)sendbuf.ptr(),
                                    comp);
    }

    ucs_status_t put_bcopy(const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf,
                           uct_completion_t *comp)
    {
        ssize_t packed_len;
        packed_len = uct_ep_put_bcopy(sender().ep(0), mapped_buffer::pack,
                                      (void*)&sendbuf, recvbuf.addr(), recvbuf.rkey());
        if (packed_len >= 0) {
            EXPECT_EQ(sendbuf.length(), (size_t)packed_len);
            return UCS_OK;
        } else {
            return (ucs_status_t)packed_len;
        }
    }

    ucs_status_t am_short(const mapped_buffer &sendbuf,
                          const mapped_buffer &recvbuf,
                          uct_completion_t *comp)
    {
        ucs_status_t status;
        status = uct_ep_am_short(sender().ep(0), AM_ID,
                                 *(uint64_t*)sendbuf.ptr(),
                                 (uint64_t*)sendbuf.ptr() + 1,
                                 sendbuf.length() - sizeof(uint64_t));
        if (status == UCS_OK) {
            ucs_atomic_add32(&am_pending, +1);
        }
        return status;
    }

    ucs_status_t am_zcopy(const mapped_buffer &sendbuf,
                          const mapped_buffer &recvbuf,
                          uct_completion_t *comp)
    {
        ucs_status_t status;
        size_t header_length;
        uct_iov_t iov;

        header_length = ucs_min(ucs::rand() % sender().iface_attr().cap.am.max_hdr,
                                sendbuf.length());

        iov.buffer = (char*)sendbuf.ptr() + header_length;
        iov.count  = 1;
        iov.length = sendbuf.length() - header_length;
        iov.memh   = sendbuf.memh();
        status = uct_ep_am_zcopy(sender().ep(0), AM_ID, sendbuf.ptr(), header_length,
                                 &iov, 1, 0, comp);
        if (status == UCS_OK || status == UCS_INPROGRESS) {
            ucs_atomic_add32(&am_pending, +1);
        }
        return status;
    }

    void random_op(const mapped_buffer &sendbuf, const mapped_buffer &recvbuf)
    {
        uct_completion_t comp;
        ucs_status_t status;
        int op;

        op         = ucs::rand() % m_avail_send_funcs.size();
        comp.count = 1;
        comp.func  = completion_callback;

        for (;;) {
            status = (this->*m_avail_send_funcs[op])(sendbuf, recvbuf, &comp);
            if (status == UCS_OK) {
                break;
            } else if (status == UCS_INPROGRESS) {
                /* coverity[loop_condition] */
                while (comp.count > 0) {
                    progress();
                }
                break;
            } else if (status == UCS_ERR_NO_RESOURCE) {
                progress();
                continue;
            } else {
                ASSERT_UCS_OK(status);
            }
        }
    }

protected:
    virtual void init() {
        uct_p2p_test::init();
        ucs_status_t status = uct_iface_set_am_handler(receiver().iface(), AM_ID,
                                                       am_callback, NULL,
                                                       UCT_AM_CB_FLAG_ASYNC);
        ASSERT_UCS_OK(status);

        m_send_size = MAX_SIZE;
        if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
            m_avail_send_funcs.push_back(&uct_p2p_mix_test::am_short);
            m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.am.max_short);
        }
        if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
            m_avail_send_funcs.push_back(&uct_p2p_mix_test::am_zcopy);
            m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.am.max_zcopy);
        }
        if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
            m_avail_send_funcs.push_back(&uct_p2p_mix_test::put_bcopy);
            m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.put.max_bcopy);
        }
        if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_ATOMIC_SWAP64) {
            m_avail_send_funcs.push_back(&uct_p2p_mix_test::swap64);
        }
        if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_ATOMIC_CSWAP64) {
            m_avail_send_funcs.push_back(&uct_p2p_mix_test::cswap64);
        }
        if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_ATOMIC_SWAP32) {
            m_avail_send_funcs.push_back(&uct_p2p_mix_test::swap32);
        }
        if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_ATOMIC_FADD32) {
            m_avail_send_funcs.push_back(&uct_p2p_mix_test::fadd32);
        }
    }

    virtual void cleanup() {
        while (am_pending) {
            progress();
        }
        uct_iface_set_am_handler(receiver().iface(), AM_ID, NULL, NULL, 0);
        uct_p2p_test::cleanup();
    }

    std::vector<send_func_t> m_avail_send_funcs;
    size_t                   m_send_size;
    static uint32_t          am_pending;
};

uint32_t uct_p2p_mix_test::am_pending = 0;

UCS_TEST_P(uct_p2p_mix_test, mix1) {

    if (m_avail_send_funcs.size() == 0) {
        UCS_TEST_SKIP_R("unsupported");
    }

    mapped_buffer sendbuf(m_send_size, 0, sender());
    mapped_buffer recvbuf(m_send_size, 0, receiver());

    for (int i = 0; i < 10000; ++i) {
        random_op(sendbuf, recvbuf);
    }
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_mix_test)
