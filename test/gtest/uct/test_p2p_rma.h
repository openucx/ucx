/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_TEST_P2P_RMA
#define UCT_TEST_P2P_RMA

#include "uct_p2p_test.h"

class uct_p2p_rma_test : public uct_p2p_test {
public:
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    static const uint64_t SEED3 = 0x3333333333333333lu;

    uct_p2p_rma_test();

    ucs_status_t put_short(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf);

    ucs_status_t put_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf);

    ucs_status_t put_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf);

    ucs_status_t get_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf);

    ucs_status_t get_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf);

    virtual void test_xfer(send_func_t send, size_t length,
                           unsigned flags, uct_memory_type_t mem_type);
};

#endif
