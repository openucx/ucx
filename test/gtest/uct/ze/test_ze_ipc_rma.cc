/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>

extern "C" {
#include <uct/ze/base/ze_base.h>
}


/*
 * Endpoint-level RMA tests for ze_ipc. Both entities live in this process,
 * so the transport takes its same-pid loopback path (no pidfd_getfd fd
 * duplication), but the iface open/connect/put_zcopy/get_zcopy/flush and
 * cache map/unmap paths are exercised the same way they would be across
 * processes.
 */
class test_ze_ipc_rma : public uct_test {
protected:
    void init()
    {
        uct_test::init();

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_sender->connect(0, *m_receiver, 0);
    }

    entity *m_sender;
    entity *m_receiver;

    static const uint64_t SEED1 = 0xABClu;
    static const uint64_t SEED2 = 0xDEFlu;
};


UCS_TEST_P(test_ze_ipc_rma, put_zcopy)
{
    size_t length = 4096;

    mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_put_zcopy(m_sender->ep(0), sendbuf.iov(),
                                                 1, recvbuf.addr(),
                                                 recvbuf.rkey(), NULL));
    m_sender->flush();
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_P(test_ze_ipc_rma, get_zcopy)
{
    size_t length = 4096;

    mapped_buffer sendbuf(length, SEED1, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_get_zcopy(m_sender->ep(0), recvbuf.iov(),
                                                 1, sendbuf.addr(),
                                                 sendbuf.rkey(), NULL));
    m_sender->flush();
    recvbuf.pattern_check(SEED1);
}

UCS_TEST_P(test_ze_ipc_rma, put_zcopy_rejects_multi_iov)
{
    size_t length = 64;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    uct_iov_t iov[2];

    iov[0] = *sendbuf.iov();
    iov[1] = *sendbuf.iov();

    ucs_status_t status;
    {
        scoped_log_handler slh(hide_errors_logger);
        status = uct_ep_put_zcopy(m_sender->ep(0), iov, 2, recvbuf.addr(),
                                  recvbuf.rkey(), NULL);
    }
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_ze_ipc_rma, put_zcopy_rejects_out_of_range_rkey)
{
    size_t length = 64;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0,
                          UCS_MEMORY_TYPE_ZE_DEVICE);

    ucs_status_t status;
    {
        /* the rkey covers the whole driver allocation, which is page-rounded
         * and therefore larger than 'length' - go far enough past the base
         * address to land outside it */
        scoped_log_handler slh(hide_errors_logger);
        status = uct_ep_put_zcopy(m_sender->ep(0), sendbuf.iov(), 1,
                                  recvbuf.addr() + UCS_MBYTE, recvbuf.rkey(),
                                  NULL);
    }
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

_UCT_INSTANTIATE_TEST_CASE(test_ze_ipc_rma, ze_ipc)
