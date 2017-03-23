/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
}
#include <common/test.h>
#include "uct_test.h"


class test_tag : public uct_test
{
public:
    static const uint64_t SEND_SEED  = 0xa1a1a1a1a1a1a1a1ul;
    static const uint64_t RECV_SEED  = 0xb2b2b2b2b2b2b2b2ul;
    static const uint64_t MASK       = 0xfffffffffffffffful;

    struct recv_ctx {
        recv_ctx(mapped_buffer *b, uct_tag_t t, uct_tag_t m = MASK) :
                 mbuf(b), tag(t), tmask(m) {

            uct_ctx.completed_cb    = completed;
            uct_ctx.tag_consumed_cb = tag_consumed;
            comp = unexp = consumed = false;
        }

        mapped_buffer     *mbuf;
        uct_tag_t         tag;
        uct_tag_t         tmask;
        bool              comp;
        bool              unexp;
        bool              consumed;
        uct_tag_context_t uct_ctx;
        ucs_status_t      status;
    };

    struct send_ctx {
        send_ctx(mapped_buffer *b, uct_tag_t t, uint64_t i) :
                 mbuf(b), tag(t), imm_data(i) {

            uct_comp.count = 2;
            uct_comp.func = NULL;
        }
        mapped_buffer    *mbuf;
        uct_tag_t        tag;
        uint64_t         imm_data;
        uct_completion_t uct_comp;
    };

    typedef ucs_status_t (test_tag::*send_func)(entity&, send_ctx&);

    void init() {
        uct_test::init();

        uct_iface_params params;

        // tl and dev names are taken from resources via GetParam, no need
        // to fill it here
        params.rx_headroom = 0;
        params.eager_cb    = unexp_eager;
        params.eager_arg   = NULL;
        params.rndv_cb     = unexp_rndv;
        params.rndv_arg    = NULL;

        if (UCT_DEVICE_TYPE_SELF == GetParam()->dev_type) {
            entity *e = uct_test::create_entity(params);
            m_entities.push_back(e);

            e->connect(0, *e, 0);
        } else {
            entity *m_sender = uct_test::create_entity(params);
            m_entities.push_back(m_sender);

            entity *m_receiver = uct_test::create_entity(params);
            m_entities.push_back(m_receiver);

            m_sender->connect(0, *m_receiver, 0);
        }
    }

    void wait_completion (bool *event) {
        ucs_time_t loop_limit = ucs_get_time() +
                                ucs_time_from_sec(DEFAULT_TIMEOUT_SEC);
        while (!*event) {
            if (ucs_get_time() > loop_limit) {
                break;
            }
            progress();
        }
    }

    ucs_status_t tag_eager_short(entity &e, send_ctx &ctx)
    {
        return uct_ep_tag_eager_short(e.ep(0), ctx.tag, ctx.mbuf->ptr(),
                                      ctx.mbuf->length());
    }

    ucs_status_t tag_eager_bcopy(entity &e, send_ctx &ctx)
    {
        ssize_t status = uct_ep_tag_eager_bcopy(e.ep(0), ctx.tag,
                                                ctx.imm_data, mapped_buffer::pack,
                                                reinterpret_cast<void*>(ctx.mbuf));

        return (status >= 0) ? UCS_OK : static_cast<ucs_status_t>(status);
    }

    ucs_status_t tag_eager_zcopy(entity &e, send_ctx &ctx)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, ctx.mbuf->ptr(),
                                ctx.mbuf->length(), ctx.mbuf->memh(),
                                sender().iface_attr().cap.tag.eager.max_iov);

        ucs_status_t status = uct_ep_tag_eager_zcopy(e.ep(0), ctx.tag, ctx.imm_data,
                                                     iov, iovcnt, &ctx.uct_comp);
        if (status == UCS_INPROGRESS) {
            status = UCS_OK;
        }
        return status;
    }

    ucs_status_t tag_rndv_zcopy(entity &e, send_ctx &ctx)
    {
         uint64_t ctxs[2] = {ctx.imm_data, reinterpret_cast<uint64_t>(&ctx)};

         UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, ctx.mbuf->ptr(),
                                 ctx.mbuf->length(), ctx.mbuf->memh(), 1);

         ucs_status_ptr_t status =  uct_ep_tag_rndv_zcopy(e.ep(0), ctx.tag,
                                                          &ctxs, sizeof(ctxs),
                                                          iov, iovcnt,
                                                          &ctx.uct_comp);

         return  (UCS_PTR_IS_ERR(status)) ? UCS_PTR_STATUS(status) : UCS_OK;
    }

    ucs_status_t tag_post(entity &e, recv_ctx &ctx)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, ctx.mbuf->ptr(),
                                ctx.mbuf->length(), ctx.mbuf->memh(), 1);
        return uct_iface_tag_recv_zcopy(e.iface(), ctx.tag, ctx.tmask,
                                        iov, iovcnt, &ctx.uct_ctx);
    }

    ucs_status_t tag_cancel(entity &e, recv_ctx &ctx, int force)
    {
        return uct_iface_tag_recv_cancel(e.iface(), &ctx.uct_ctx, force);
    }

    // If expected message arrives, two callbacks should be called:
    // tag_consumed and completed (unexpected callback should not be
    // called). And it is vice versa if message arrives unexpectedly
    void check_completion(recv_ctx &ctx, bool is_expected, uint64_t seed,
                          ucs_status_t status = UCS_OK) {
      EXPECT_EQ(ctx.comp,     is_expected);
      EXPECT_EQ(ctx.consumed, is_expected);
      EXPECT_EQ(ctx.unexp,    !is_expected);
      EXPECT_EQ(ctx.status,   status);
      ctx.mbuf->pattern_check(seed);
    }

    void test_eager_expected(send_func sfunc) {
        const size_t length = 1024;
        uct_tag_t    tag    = 11;

        mapped_buffer recvbuf(length, RECV_SEED, receiver());
        recv_ctx r_ctx(&recvbuf, tag);
        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        short_progress_loop();

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        send_ctx s_ctx(&sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        wait_completion(&r_ctx.comp);

        check_completion(r_ctx, true, SEND_SEED);

        short_progress_loop();
    }

    void test_eager_unexpected(send_func sfunc) {
        const size_t length = 65;
        uct_tag_t    tag    = 11;

        mapped_buffer recvbuf(length, RECV_SEED, receiver());
        mapped_buffer sendbuf(length, SEND_SEED, sender());
        recv_ctx r_ctx(&recvbuf, tag);
        send_ctx s_ctx(&sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        short_progress_loop();

        wait_completion(&r_ctx.unexp);

        check_completion(r_ctx, false, SEND_SEED);
    }

    void test_eager_wrong_tag(send_func sfunc) {
        const size_t length = 65;
        uct_tag_t    tag    = 11;

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        mapped_buffer recvbuf(length, RECV_SEED, receiver());

        // Post modified tag for incoming message to be reported as unexpected
        // and not to be macthed.
        recv_ctx r_ctx(&recvbuf, tag + 1);
        send_ctx s_ctx(&sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));

        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        short_progress_loop();

        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        wait_completion(&r_ctx.unexp);

        // Message should be reported as unexpected and filled with
        // recv seed (unchanged), as the incoming tag does not match the expected
        check_completion(r_ctx, false, RECV_SEED);
    }

    void test_eager_tag_mask(send_func sfunc) {
        const size_t length = 65;

        mapped_buffer recvbuf(length, RECV_SEED, receiver());

        // Post tag and tag mask in a way that it matches sender tag with
        // tag_mask applied, but is not exactly the same.
        recv_ctx r_ctx(&recvbuf, 0xff, 0xff);
        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        short_progress_loop();

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        send_ctx s_ctx(&sendbuf, 0xffff, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));
        wait_completion(&r_ctx.comp);

        // Should be matched because tags are equal with tag mask applied.
        check_completion(r_ctx, true, SEND_SEED);
    }

    static void tag_consumed(uct_tag_context_t *self)
    {
        recv_ctx *user_ctx = ucs_container_of(self, recv_ctx, uct_ctx);
        user_ctx->consumed = true;
    }

    static void completed(uct_tag_context_t *self, uct_tag_t stag, uint64_t imm,
                          size_t length, ucs_status_t status)
    {
        recv_ctx *user_ctx = ucs_container_of(self, recv_ctx, uct_ctx);
        user_ctx->comp     = true;
        EXPECT_EQ(user_ctx->tag, (stag & user_ctx->tmask));
        EXPECT_EQ(user_ctx->mbuf->length(), length);
        user_ctx->status = status;
    }

    static ucs_status_t unexp_eager(void *arg, void *data, size_t length,
                                    void *desc, uct_tag_t stag, uint64_t imm)
    {
        recv_ctx *user_ctx = reinterpret_cast<recv_ctx*>(imm);
        user_ctx->unexp    = true;
        user_ctx->status   = UCS_OK;
        if (user_ctx->tag == stag) {
            memcpy(user_ctx->mbuf->ptr(), data, ucs_min(length,
                   user_ctx->mbuf->length()));
        }
        return UCS_OK;
    }

    static ucs_status_t unexp_rndv(void *arg, void *desc, uint64_t stag,
                                   const void *header, unsigned header_length,
                                   uint64_t remote_addr, size_t length,
                                   const void *rkey_buf)
    {
        uint64_t *ctxs  = const_cast<uint64_t*>(static_cast<const uint64_t*>(header));
        recv_ctx *r_ctx = reinterpret_cast<recv_ctx*>(*ctxs);
        send_ctx *s_ctx = reinterpret_cast<send_ctx*>(*(ctxs + 1));

        r_ctx->unexp  = true;
        r_ctx->status = UCS_OK;

        EXPECT_EQ(s_ctx->tag, stag);
        EXPECT_EQ(s_ctx->mbuf->length(), length);
        EXPECT_EQ(reinterpret_cast<uint64_t>(s_ctx->mbuf->ptr()), remote_addr);

        return UCS_OK;
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length, void *desc) {
        is_am_received = true;
        return UCS_OK;
    }

protected:
    uct_test::entity& sender() {
        return **m_entities.begin();
    }

    uct_test::entity& receiver() {
        return **(m_entities.end() - 1);
    }

    static bool is_am_received;
};

bool test_tag::is_am_received = false;

UCS_TEST_P(test_tag, tag_eager_short_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT);
    test_eager_expected(static_cast<send_func>(&test_tag::tag_eager_short));
}

UCS_TEST_P(test_tag, tag_eager_bcopy_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_eager_expected(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_P(test_tag, tag_eager_zcopy_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_eager_expected(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_P(test_tag, tag_rndv_zcopy_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
    test_eager_expected(static_cast<send_func>(&test_tag::tag_rndv_zcopy));
}

UCS_TEST_P(test_tag, tag_eager_bcopy_unexpected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_eager_unexpected(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_P(test_tag, tag_eager_zcopy_unexpected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_eager_unexpected(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_P(test_tag, tag_rndv_zcopy_unexpected)
{
    check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
    test_eager_unexpected(static_cast<send_func>(&test_tag::tag_rndv_zcopy));
}

UCS_TEST_P(test_tag, tag_eager_bcopy_wrong_tag)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_eager_wrong_tag(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_P(test_tag, tag_eager_zcopy_wrong_tag)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_eager_wrong_tag(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_P(test_tag, tag_eager_short_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT);
    test_eager_tag_mask(static_cast<send_func>(&test_tag::tag_eager_short));
}

UCS_TEST_P(test_tag, tag_eager_bcopy_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_eager_tag_mask(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_P(test_tag, tag_eager_zcopy_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_eager_tag_mask(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_P(test_tag, tag_rndv_zcopy_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
    test_eager_tag_mask(static_cast<send_func>(&test_tag::tag_rndv_zcopy));
}

UCS_TEST_P(test_tag, tag_send_no_tag)
{
  check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);

  uct_iface_set_am_handler(receiver().iface(), 0, am_handler,
                           NULL, UCT_AM_CB_FLAG_SYNC);
  mapped_buffer lbuf(200, SEND_SEED, sender());
  ssize_t len = uct_ep_am_bcopy(sender().ep(0), 0, mapped_buffer::pack,
                                reinterpret_cast<void*>(&lbuf));
  EXPECT_EQ(lbuf.length(), len);
  short_progress_loop();
  EXPECT_TRUE(is_am_received);
}

UCS_TEST_P(test_tag, tag_cancel_force)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);

    const size_t length = 128;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx(&recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));
    short_progress_loop(200);
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 1));

    short_progress_loop();

    mapped_buffer sendbuf(length, SEND_SEED, sender());
    send_ctx s_ctx(&sendbuf, 1, reinterpret_cast<uint64_t>(&r_ctx));
    ASSERT_UCS_OK(tag_eager_bcopy(sender(), s_ctx));

    // Message should arrive unexpected, since tag was cancelled
    // on the receiver.
    wait_completion(&r_ctx.unexp);
    check_completion(r_ctx, false, SEND_SEED);
}

UCS_TEST_P(test_tag, tag_cancel_noforce)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    const size_t length = 128;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx(&recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));
    short_progress_loop(200);
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 0));

    short_progress_loop();

    // Check that completed callback has been called with CANCELED status
    // (because 0 was passed as force parameter to cancel).
    EXPECT_TRUE(r_ctx.comp);
    EXPECT_EQ(r_ctx.status, UCS_ERR_CANCELED);
}

UCS_TEST_P(test_tag, tag_limit)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    const size_t length = 32;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    ucs::ptr_vector<recv_ctx> rctxs;
    recv_ctx *rctx_p;
    ucs_status_t status;

    do {
        // Can use the same recv buffer, as no sends will be issued.
        rctx_p = (new recv_ctx(&recvbuf, 1));
        rctxs.push_back(rctx_p);
        status = tag_post(receiver(), *rctx_p);
        // Make sure send resources are acknowledged, as we
        // awaiting for tag space exhaustion.
        short_progress_loop();
    } while (status == UCS_OK);

    EXPECT_EQ(status, UCS_ERR_EXCEEDS_LIMIT);

    // Cancel one of the postings
    ASSERT_UCS_OK(tag_cancel(receiver(), rctxs.at(0), 1));
    short_progress_loop();

    // Check we can post again
    ASSERT_UCS_OK(tag_post(receiver(), rctxs.at(0)));
}

_UCT_INSTANTIATE_TEST_CASE(test_tag, rc)


