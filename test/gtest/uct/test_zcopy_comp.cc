/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_test.h"


class test_zcopy_comp : public uct_test {
};


UCS_TEST_P(test_zcopy_comp, issue1440)
{
    entity *sender = create_entity(0);
    m_entities.push_back(sender);

    entity *receiver_small = create_entity(0);
    m_entities.push_back(receiver_small);

    entity *receiver_large = create_entity(0);
    m_entities.push_back(receiver_large);

    sender->connect(0, *receiver_small, 0);
    sender->connect(1, *receiver_large, 0);

    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY);

    size_t size_small = ucs_max(8ul,     sender->iface_attr().cap.put.min_zcopy);
    size_t size_large = ucs_min(65536ul, sender->iface_attr().cap.put.max_zcopy);
    ucs_assert(size_large > size_small);

    mapped_buffer sendbuf_small(size_small, 0, *sender);
    mapped_buffer sendbuf_large(size_large, 0, *sender);
    mapped_buffer recvbuf_small(size_small, 0, *receiver_small);
    mapped_buffer recvbuf_large(size_large, 0, *receiver_large);

    /*
     * Send a mix of small messages to one destination and large messages to
     * another destination. This can trigger overriding RC/DC send completions.
     */
    uct_completion_t dummy_comp = { NULL, INT_MAX };
    int num_small_sends = 1000000 / ucs::test_time_multiplier();
    int num_large_sends = 1000 / ucs::test_time_multiplier();
    while (num_small_sends || num_large_sends) {
        if (num_small_sends) {
            ucs_status_t status;
            status = uct_ep_put_zcopy(sender->ep(0), sendbuf_small.iov(), 1,
                                      (uintptr_t)recvbuf_small.ptr(),
                                      recvbuf_small.rkey(), &dummy_comp);
            if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
                --num_small_sends;
            }
        }
        if (num_large_sends) {
            ucs_status_t status;
            status = uct_ep_put_zcopy(sender->ep(1), sendbuf_large.iov(), 1,
                                      (uintptr_t)recvbuf_large.ptr(),
                                      recvbuf_large.rkey(), &dummy_comp);
            if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
                --num_large_sends;
            }
        }
        progress();
    }

    sender->flush();
}


UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_zcopy_comp)
