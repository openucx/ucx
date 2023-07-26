/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_p2p_mix.h"

extern "C" {
#include <ucs/arch/atomic.h>
}
#include <functional>


uct_p2p_mix_test::uct_p2p_mix_test() : uct_p2p_test(0), m_send_size(0) {
}

ucs_status_t uct_p2p_mix_test::am_callback(void *arg, void *data, size_t length,
                                           unsigned flags)
{
    ucs_atomic_sub32(&am_pending, 1);
    return UCS_OK;
}

void uct_p2p_mix_test::completion_callback(uct_completion_t *comp)
{
    EXPECT_UCS_OK(comp->status);
}

template <typename T, uct_atomic_op_t OP>
ucs_status_t uct_p2p_mix_test::uct_p2p_mix_test::atomic_fop(const mapped_buffer &sendbuf,
                                                            const mapped_buffer &recvbuf,
                                                            uct_completion_t *comp)
{
    if (sizeof(T) == sizeof(uint32_t)) {
        return uct_ep_atomic32_fetch(sender().ep(0), OP, 1, (uint32_t*)sendbuf.ptr(),
                                     recvbuf.addr(), recvbuf.rkey(), comp);
    } else {
        return uct_ep_atomic64_fetch(sender().ep(0), OP, 1, (uint64_t*)sendbuf.ptr(),
                                     recvbuf.addr(), recvbuf.rkey(), comp);
    }
}

ucs_status_t uct_p2p_mix_test::cswap64(const mapped_buffer &sendbuf,
                                       const mapped_buffer &recvbuf,
                                       uct_completion_t *comp)
{
    return uct_ep_atomic_cswap64(sender().ep(0), 0, 1, recvbuf.addr(),
                                 recvbuf.rkey(), (uint64_t*)sendbuf.ptr(),
                                 comp);
}

ucs_status_t uct_p2p_mix_test::put_short(const mapped_buffer &sendbuf,
                                         const mapped_buffer &recvbuf,
                                         uct_completion_t *comp)
{
    return uct_ep_put_short(sender().ep(0), sendbuf.ptr(),
                            sendbuf.length(), recvbuf.addr(),
                            recvbuf.rkey());
}

ucs_status_t uct_p2p_mix_test::put_bcopy(const mapped_buffer &sendbuf,
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

ucs_status_t uct_p2p_mix_test::am_short(const mapped_buffer &sendbuf,
                                        const mapped_buffer &recvbuf,
                                        uct_completion_t *comp)
{
    ucs_status_t status;
    status = uct_ep_am_short(sender().ep(0), AM_ID, *(uint64_t*)sendbuf.ptr(),
                             (uint64_t*)sendbuf.ptr() + 1,
                             sendbuf.length() - sizeof(uint64_t));
    if (status == UCS_OK) {
        ucs_atomic_add32(&am_pending, +1);
    }
    return status;
}

ucs_status_t uct_p2p_mix_test::am_short_iov(const mapped_buffer &sendbuf,
                                            const mapped_buffer &recvbuf,
                                            uct_completion_t *comp)
{
    ucs_status_t status;
    uct_iov_t iov;

    iov.buffer = sendbuf.ptr();
    iov.length = sendbuf.length();
    iov.count  = 1;
    iov.stride = 0;
    iov.memh   = sendbuf.memh();

    status = uct_ep_am_short_iov(sender().ep(0), AM_ID, &iov, 1);
    if (status == UCS_OK) {
        ucs_atomic_add32(&am_pending, +1);
    }
    return status;
}

ucs_status_t uct_p2p_mix_test::am_zcopy(const mapped_buffer &sendbuf,
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

void uct_p2p_mix_test::random_op(const mapped_buffer &sendbuf,
                                 const mapped_buffer &recvbuf)
{
    uct_completion_t comp;
    ucs_status_t status;
    int op;

    op          = ucs::rand() % m_avail_send_funcs.size();
    comp.count  = 1;
    comp.status = UCS_OK;
    comp.func   = completion_callback;

    for (;;) {
        status = (this->*m_avail_send_funcs[op])(sendbuf, recvbuf, &comp);
        if (status == UCS_INPROGRESS) {
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
            break;
        }
    }
}

void uct_p2p_mix_test::run(unsigned count) {
    if (m_avail_send_funcs.size() == 0) {
        UCS_TEST_SKIP_R("unsupported");
    }
    if (!(sender().md_attr().access_mem_types & UCS_BIT(UCS_MEMORY_TYPE_HOST))) {
        UCS_TEST_SKIP_R("skipping on non-host memory");
    }

    mapped_buffer sendbuf(m_send_size, 0, sender());
    mapped_buffer recvbuf(m_send_size, 0, receiver());

    for (unsigned i = 0; i < count; ++i) {
        random_op(sendbuf, recvbuf);
    }

    flush();
}

void uct_p2p_mix_test::init() {
    uct_p2p_test::init();
    ucs_status_t status = uct_iface_set_am_handler(receiver().iface(), AM_ID,
                                                   am_callback, NULL,
                                                   UCT_CB_FLAG_ASYNC);
    ASSERT_UCS_OK(status);

    m_send_size = MAX_SIZE;
    if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::am_short);
        m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.am.max_short);

        m_avail_send_funcs.push_back(&uct_p2p_mix_test::am_short_iov);
        m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.am.max_short);
    }
    if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::am_zcopy);
        m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.am.max_zcopy);
    }
    if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_PUT_SHORT) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::put_short);
        m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.put.max_short);
    }
    if (sender().iface_attr().cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::put_bcopy);
        m_send_size = ucs_min(m_send_size, sender().iface_attr().cap.put.max_bcopy);
    }
    if (sender().iface_attr().cap.atomic64.fop_flags & UCS_BIT(UCT_ATOMIC_OP_CSWAP)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::cswap64);
    }
    if (sender().iface_attr().cap.atomic64.fop_flags & UCS_BIT(UCT_ATOMIC_OP_ADD)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint64_t, UCT_ATOMIC_OP_ADD>);
    }
    if (sender().iface_attr().cap.atomic32.fop_flags & UCS_BIT(UCT_ATOMIC_OP_ADD)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint32_t, UCT_ATOMIC_OP_ADD>);
    }
    if (sender().iface_attr().cap.atomic64.fop_flags & UCS_BIT(UCT_ATOMIC_OP_AND)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint64_t, UCT_ATOMIC_OP_AND>);
    }
    if (sender().iface_attr().cap.atomic32.fop_flags & UCS_BIT(UCT_ATOMIC_OP_AND)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint32_t, UCT_ATOMIC_OP_AND>);
    }
    if (sender().iface_attr().cap.atomic64.fop_flags & UCS_BIT(UCT_ATOMIC_OP_OR)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint64_t, UCT_ATOMIC_OP_OR>);
    }
    if (sender().iface_attr().cap.atomic32.fop_flags & UCS_BIT(UCT_ATOMIC_OP_OR)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint32_t, UCT_ATOMIC_OP_OR>);
    }
    if (sender().iface_attr().cap.atomic64.fop_flags & UCS_BIT(UCT_ATOMIC_OP_XOR)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint64_t, UCT_ATOMIC_OP_XOR>);
    }
    if (sender().iface_attr().cap.atomic32.fop_flags & UCS_BIT(UCT_ATOMIC_OP_XOR)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint32_t, UCT_ATOMIC_OP_XOR>);
    }
    if (sender().iface_attr().cap.atomic64.fop_flags & UCS_BIT(UCT_ATOMIC_OP_SWAP)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint64_t, UCT_ATOMIC_OP_SWAP>);
    }
    if (sender().iface_attr().cap.atomic32.fop_flags & UCS_BIT(UCT_ATOMIC_OP_SWAP)) {
        m_avail_send_funcs.push_back(&uct_p2p_mix_test::atomic_fop<uint32_t, UCT_ATOMIC_OP_SWAP>);
    }
}

void uct_p2p_mix_test::cleanup() {
    while (am_pending) {
        progress();
    }
    uct_iface_set_am_handler(receiver().iface(), AM_ID, NULL, NULL, 0);
    uct_p2p_test::cleanup();
}

uint32_t uct_p2p_mix_test::am_pending = 0;

UCS_TEST_P(uct_p2p_mix_test, mix_10000) {
    run(10000);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_mix_test)
