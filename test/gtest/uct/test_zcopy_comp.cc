/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_test.h"


class test_zcopy_comp : public uct_test {
protected:
    virtual void init() {
        uct_test::init();

        m_sender = create_entity(0);
        m_entities.push_back(m_sender);

        check_skip_test();
    }

protected:
    entity *m_sender;
};


UCS_TEST_SKIP_COND_P(test_zcopy_comp, issue1440,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    entity *receiver_small = create_entity(0);
    m_entities.push_back(receiver_small);

    entity *receiver_large = create_entity(0);
    m_entities.push_back(receiver_large);

    m_sender->connect(0, *receiver_small, 0);
    m_sender->connect(1, *receiver_large, 0);

    size_t size_small = ucs_max(8ul,     m_sender->iface_attr().cap.put.min_zcopy);
    size_t size_large = ucs_min(65536ul, m_sender->iface_attr().cap.put.max_zcopy);
    ucs_assert(size_large > size_small);

    if (!(m_sender->md_attr().cap.access_mem_types & UCS_BIT(UCS_MEMORY_TYPE_HOST))) {
        std::stringstream ss;
        ss << "test_zcopy_comp is not supported by " << GetParam();
        UCS_TEST_SKIP_R(ss.str());
    }

    mapped_buffer sendbuf_small(size_small, 0, *m_sender);
    mapped_buffer sendbuf_large(size_large, 0, *m_sender);
    mapped_buffer recvbuf_small(size_small, 0, *receiver_small);
    mapped_buffer recvbuf_large(size_large, 0, *receiver_large);

    /*
     * Send a mix of small messages to one destination and large messages to
     * another destination. This can trigger overriding RC/DC send completions.
     */
    uct_completion_t dummy_comp = { NULL, INT_MAX, UCS_OK };
    int num_small_sends = 1000000 / ucs::test_time_multiplier();
    int num_large_sends = 1000 / ucs::test_time_multiplier();
    while (num_small_sends || num_large_sends) {
        if (num_small_sends) {
            ucs_status_t status;
            status = uct_ep_put_zcopy(m_sender->ep(0), sendbuf_small.iov(), 1,
                                      (uintptr_t)recvbuf_small.ptr(),
                                      recvbuf_small.rkey(), &dummy_comp);
            if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
                --num_small_sends;
            }
        }
        if (num_large_sends) {
            ucs_status_t status;
            status = uct_ep_put_zcopy(m_sender->ep(1), sendbuf_large.iov(), 1,
                                      (uintptr_t)recvbuf_large.ptr(),
                                      recvbuf_large.rkey(), &dummy_comp);
            if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
                --num_large_sends;
            }
        }
        progress();
    }

    /* Call flush on local and remote ifaces to progress data
     * (e.g. if call flush only on local iface, a target side may
     *  not be able to send PUT ACK to an initiator in case of TCP) */
    flush();
}


UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_zcopy_comp)
