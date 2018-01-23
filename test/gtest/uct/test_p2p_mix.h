/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_TEST_P2P_MIX_H
#define UCT_TEST_P2P_MIX_H

#include "uct_p2p_test.h"

class uct_p2p_mix_test : public uct_p2p_test {
public:

    typedef ucs_status_t
            (uct_p2p_mix_test::* send_func_t)(const mapped_buffer &sendbuf,
                                              const mapped_buffer &recvbuf,
                                              uct_completion_t *comp);

    static const uint8_t AM_ID    = 1;
    static const size_t  MAX_SIZE = 256;

    uct_p2p_mix_test();

protected:
    static ucs_status_t am_callback(void *arg, void *data, size_t length,
                                    unsigned flags);

    static void completion_callback(uct_completion_t *comp, ucs_status_t status);

    ucs_status_t swap64(const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf,
                        uct_completion_t *comp);

    ucs_status_t cswap64(const mapped_buffer &sendbuf,
                         const mapped_buffer &recvbuf,
                         uct_completion_t *comp);

    ucs_status_t fadd32(const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf,
                        uct_completion_t *comp);

    ucs_status_t swap32(const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf,
                        uct_completion_t *comp);

    ucs_status_t put_short(const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf,
                           uct_completion_t *comp);

    ucs_status_t put_bcopy(const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf,
                           uct_completion_t *comp);

    ucs_status_t am_short(const mapped_buffer &sendbuf,
                          const mapped_buffer &recvbuf,
                          uct_completion_t *comp);

    ucs_status_t am_zcopy(const mapped_buffer &sendbuf,
                          const mapped_buffer &recvbuf,
                          uct_completion_t *comp);

    void random_op(const mapped_buffer &sendbuf, const mapped_buffer &recvbuf);

    void run(unsigned count);

    virtual void init();

    virtual void cleanup();

private:
    std::vector<send_func_t> m_avail_send_funcs;
    size_t                   m_send_size;
    static uint32_t          am_pending;
};

#endif
