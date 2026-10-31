/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_p2p_rma.h"

class test_musa_copy : public uct_p2p_rma_test {
public:
    static std::vector<const resource*>
    enum_resources(const std::string &tl_name)
    {
        std::vector<const resource*> resources =
                uct_p2p_rma_test::enum_resources(tl_name);
        std::vector<const resource*> result;

        for (const resource *rsc : resources) {
            const p2p_resource *p2p_rsc =
                    dynamic_cast<const p2p_resource*>(rsc);
            if ((p2p_rsc != NULL) && p2p_rsc->loopback) {
                result.push_back(rsc);
            }
        }

        return result;
    }
};

UCS_TEST_SKIP_COND_P(test_musa_copy, put_zcopy_is_async,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    mapped_buffer sendbuf(4096, SEED1, sender(), 0, UCS_MEMORY_TYPE_HOST);
    mapped_buffer recvbuf(4096, SEED2, receiver(), 0, UCS_MEMORY_TYPE_MUSA);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(),
                            sender().iface_attr().cap.put.max_iov);

    ucs_status_t status = uct_ep_put_zcopy(sender_ep(), iov, iovcnt,
                                           recvbuf.addr(), recvbuf.rkey(),
                                           comp());
    ASSERT_EQ(UCS_INPROGRESS, status);

    flush();
    recvbuf.pattern_check(SEED1);
}

_UCT_INSTANTIATE_TEST_CASE(test_musa_copy, musa_copy)
