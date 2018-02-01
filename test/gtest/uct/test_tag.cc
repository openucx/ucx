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


class test_tag : public uct_test {
public:
    static const uint64_t SEND_SEED  = 0xa1a1a1a1a1a1a1a1ul;
    static const uint64_t RECV_SEED  = 0xb2b2b2b2b2b2b2b2ul;
    static const uint64_t MASK       = 0xfffffffffffffffful;

    struct rndv_hdr {
        uint64_t          priv[2];
        uint16_t          tail;
    } UCS_S_PACKED;

    struct recv_ctx {
        mapped_buffer     *mbuf;
        uct_tag_t         tag;
        uct_tag_t         tmask;
        bool              take_uct_desc;
        bool              comp;
        bool              unexp;
        bool              consumed;
        bool              sw_rndv;
        uct_tag_context_t uct_ctx;
        ucs_status_t      status;
    };

    struct send_ctx {
        send_ctx(mapped_buffer *b, uct_tag_t t, uint64_t i) :
                 mbuf(b), rndv_op(NULL), tag(t), imm_data(i) {

            uct_comp.count = 2;
            uct_comp.func  = NULL;
            sw_rndv        = false;
        }
        mapped_buffer    *mbuf;
        void             *rndv_op;
        uct_tag_t        tag;
        uint64_t         imm_data;
        uct_completion_t uct_comp;
        bool sw_rndv;
    };

    typedef ucs_status_t (test_tag::*send_func)(entity&, send_ctx&);

    void init() {
        ucs_status_t status = uct_config_modify(m_iface_config, "RC_TM_ENABLE", "y");
        ASSERT_TRUE((status == UCS_OK) || (status == UCS_ERR_NO_ELEM));

        uct_test::init();

        uct_iface_params params;

        // tl and dev names are taken from resources via GetParam, no need
        // to fill it here
        params.rx_headroom = 0;
        params.open_mode   = UCT_IFACE_OPEN_MODE_DEVICE;
        params.eager_cb    = unexp_eager;
        params.eager_arg   = reinterpret_cast<void*>(this);
        params.rndv_cb     = unexp_rndv;
        params.rndv_arg    = reinterpret_cast<void*>(this);

        if (UCT_DEVICE_TYPE_SELF == GetParam()->dev_type) {
            entity *e = uct_test::create_entity(params);
            m_entities.push_back(e);

            e->connect(0, *e, 0);
        } else {
            entity *sender = uct_test::create_entity(params);
            m_entities.push_back(sender);

            entity *receiver = uct_test::create_entity(params);
            m_entities.push_back(receiver);

            sender->connect(0, *receiver, 0);
        }
    }

    void init_recv_ctx(recv_ctx &r,  mapped_buffer *b, uct_tag_t t,
                       uct_tag_t m = MASK, bool uct_d = false) {
        r.mbuf                    = b;
        r.tag                     = t;
        r.tmask                   = m;
        r.uct_ctx.completed_cb    = completed;
        r.uct_ctx.tag_consumed_cb = tag_consumed;
        r.uct_ctx.rndv_cb         = sw_rndv_completed;
        r.take_uct_desc           = uct_d;
        r.comp = r.unexp = r.consumed = r.sw_rndv = false;
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
                                                reinterpret_cast<void*>(ctx.mbuf),
                                                0);

        return (status >= 0) ? UCS_OK : static_cast<ucs_status_t>(status);
    }

    ucs_status_t tag_eager_zcopy(entity &e, send_ctx &ctx)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, ctx.mbuf->ptr(),
                                ctx.mbuf->length(), ctx.mbuf->memh(),
                                sender().iface_attr().cap.tag.eager.max_iov);

        ucs_status_t status = uct_ep_tag_eager_zcopy(e.ep(0), ctx.tag,
                                                     ctx.imm_data, iov, iovcnt,
                                                     0, &ctx.uct_comp);
        if (status == UCS_INPROGRESS) {
            status = UCS_OK;
        }
        return status;
    }

    ucs_status_t tag_rndv_zcopy(entity &e, send_ctx &ctx)
    {
         rndv_hdr hdr = {{ctx.imm_data,
                          reinterpret_cast<uint64_t>(&ctx)
                         },
                         0xFAFA
                        };

         UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, ctx.mbuf->ptr(),
                                 ctx.mbuf->length(), ctx.mbuf->memh(), 1);

         ctx.rndv_op = uct_ep_tag_rndv_zcopy(e.ep(0), ctx.tag, &hdr,
                                             sizeof(hdr), iov, iovcnt, 0,
                                             &ctx.uct_comp);

         return  (UCS_PTR_IS_ERR(ctx.rndv_op)) ? UCS_PTR_STATUS(ctx.rndv_op) :
                                                 UCS_OK;
    }

    ucs_status_t tag_rndv_cancel(entity &e, void *op)
    {
        return uct_ep_tag_rndv_cancel(e.ep(0), op);
    }

    ucs_status_t tag_rndv_request(entity &e, send_ctx &ctx)
    {
        rndv_hdr hdr = {{ctx.imm_data,
                         reinterpret_cast<uint64_t>(&ctx)
                        },
                        0xFAFA
                       };
        ctx.sw_rndv = true;

        return uct_ep_tag_rndv_request(e.ep(0), ctx.tag, &hdr, sizeof(hdr), 0);
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
    // called). And it is vice versa if message arrives unexpectedly.
    // If expected SW RNDV request arrives tag_consumed and sw_rndv_cb
    // should be called.
    void check_completion(recv_ctx &ctx, bool is_expected, uint64_t seed,
                          ucs_status_t status = UCS_OK, bool is_sw_rndv = false) {
      EXPECT_EQ(ctx.consumed, is_expected);
      EXPECT_EQ(ctx.comp,     (is_expected && !is_sw_rndv));
      EXPECT_EQ(ctx.unexp,    (!is_expected && !is_sw_rndv));
      EXPECT_EQ(ctx.sw_rndv,  is_sw_rndv);
      EXPECT_EQ(ctx.status,   status);
      if (is_expected) {
          ctx.mbuf->pattern_check(seed);
      }
    }

    void test_tag_expected(send_func sfunc, size_t length = 75) {
        uct_tag_t    tag    = 11;

        mapped_buffer recvbuf(length, RECV_SEED, receiver());
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, tag);
        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        short_progress_loop();

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        send_ctx s_ctx(&sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        wait_for_flag(&r_ctx.comp);

        check_completion(r_ctx, true, SEND_SEED);

        flush();
    }

    void test_tag_unexpected(send_func sfunc, size_t length = 75,
                             bool take_uct_desc = false) {
        uct_tag_t tag = 11;

        mapped_buffer recvbuf(length, RECV_SEED, receiver());
        mapped_buffer sendbuf(length, SEND_SEED, sender());
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, tag, MASK, take_uct_desc);
        send_ctx s_ctx(&sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        wait_for_flag(&r_ctx.unexp);
        if (static_cast<send_func>(&test_tag::tag_rndv_zcopy) == sfunc) {
            // Need to cancel origin RNDV operation, beacuse no RNDV_COMP
            // will be received (as it arrived unexpectedly and should be
            // handled by SW).
            ASSERT_UCS_OK(tag_rndv_cancel(sender(), s_ctx.rndv_op));
        }

        check_completion(r_ctx, false, SEND_SEED);
        flush();
    }

    void test_tag_wrong_tag(send_func sfunc) {
        const size_t length = 65;
        uct_tag_t    tag    = 11;

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        mapped_buffer recvbuf(length, RECV_SEED, receiver());

        // Post modified tag for incoming message to be reported as unexpected
        // and not to be macthed.
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, tag + 1);
        send_ctx s_ctx(&sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));

        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        short_progress_loop();

        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        wait_for_flag(&r_ctx.unexp);

        // Message should be reported as unexpected and filled with
        // recv seed (unchanged), as the incoming tag does not match the expected
        check_completion(r_ctx, false, RECV_SEED);
        flush();
    }

    void test_tag_mask(send_func sfunc) {
        const size_t length = 65;

        mapped_buffer recvbuf(length, RECV_SEED, receiver());

        // Post tag and tag mask in a way that it matches sender tag with
        // tag_mask applied, but is not exactly the same.
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, 0xff, 0xff);
        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        short_progress_loop();

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        send_ctx s_ctx(&sendbuf, 0xffff, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));
        wait_for_flag(&r_ctx.comp);

        // Should be matched because tags are equal with tag mask applied.
        check_completion(r_ctx, true, SEND_SEED);
        flush();
    }

    ucs_status_t unexpected_handler(recv_ctx *ctx, void *data, unsigned flags)
    {
        if (ctx->take_uct_desc && (flags & UCT_CB_PARAM_FLAG_DESC)) {
            m_uct_descs.push_back(data);
            return UCS_INPROGRESS;
        } else {
            return UCS_OK;
        }
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
        user_ctx->status   = status;
        EXPECT_EQ(user_ctx->tag, (stag & user_ctx->tmask));
        EXPECT_EQ(user_ctx->mbuf->length(), length);
    }

    static void sw_rndv_completed(uct_tag_context_t *self, uct_tag_t stag,
                                  const void *header, unsigned header_length,
                                  ucs_status_t status)
    {
        recv_ctx *user_ctx = ucs_container_of(self, recv_ctx, uct_ctx);
        user_ctx->sw_rndv  = true;
        user_ctx->status   = status;
        EXPECT_EQ(user_ctx->tag, (stag & user_ctx->tmask));
        EXPECT_EQ(user_ctx->mbuf->length(), header_length);
    }

    static ucs_status_t unexp_eager(void *arg, void *data, size_t length,
                                    unsigned flags, uct_tag_t stag, uint64_t imm)
    {
        recv_ctx *user_ctx = reinterpret_cast<recv_ctx*>(imm);
        user_ctx->unexp    = true;
        user_ctx->status   = UCS_OK;
        if (user_ctx->tag == stag) {
            memcpy(user_ctx->mbuf->ptr(), data, ucs_min(length,
                   user_ctx->mbuf->length()));
            user_ctx->mbuf->pattern_check(SEND_SEED);
        }

        test_tag *self = reinterpret_cast<test_tag*>(arg);
        return self->unexpected_handler(user_ctx, data, flags);
    }

    static ucs_status_t unexp_rndv(void *arg, unsigned flags, uint64_t stag,
                                   const void *header, unsigned header_length,
                                   uint64_t remote_addr, size_t length,
                                   const void *rkey_buf)
    {
        rndv_hdr *rhdr  = const_cast<rndv_hdr*>(static_cast<const rndv_hdr*>(header));
        recv_ctx *r_ctx = reinterpret_cast<recv_ctx*>(rhdr->priv[0]);
        send_ctx *s_ctx = reinterpret_cast<send_ctx*>(rhdr->priv[1]);
        uint16_t  tail  = rhdr->tail;
        r_ctx->unexp  = true;
        r_ctx->status = UCS_OK;

        EXPECT_EQ(tail, 0xFAFA);
        EXPECT_EQ(s_ctx->tag, stag);
        EXPECT_EQ(length, s_ctx->sw_rndv ? 0 : s_ctx->mbuf->length());
        EXPECT_EQ(remote_addr, s_ctx->sw_rndv ? 0ul :
                  reinterpret_cast<uint64_t>(s_ctx->mbuf->ptr()));

        test_tag *self = reinterpret_cast<test_tag*>(arg);
        return self->unexpected_handler(r_ctx, const_cast<void*>(header), flags);
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {
        is_am_received = true;
        return UCS_OK;
    }

    static ucs_log_func_rc_t
    log_ep_destroy(const char *file, unsigned line, const char *function,
                   ucs_log_level_t level, const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_WARN) {
            // Ignore warnings about uncompleted operations during ep destroy
            return UCS_LOG_FUNC_RC_STOP;
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

protected:
    uct_test::entity& sender() {
        return **m_entities.begin();
    }

    uct_test::entity& receiver() {
        return **(m_entities.end() - 1);
    }

    std::vector<void*> m_uct_descs;

    static bool is_am_received;
};

bool test_tag::is_am_received = false;

UCS_TEST_P(test_tag, tag_eager_short_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT);
    test_tag_expected(static_cast<send_func>(&test_tag::tag_eager_short),
                      sender().iface_attr().cap.tag.eager.max_short);
}

UCS_TEST_P(test_tag, tag_eager_bcopy_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_tag_expected(static_cast<send_func>(&test_tag::tag_eager_bcopy),
                      sender().iface_attr().cap.tag.eager.max_bcopy);
}

UCS_TEST_P(test_tag, tag_eager_zcopy_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_tag_expected(static_cast<send_func>(&test_tag::tag_eager_zcopy),
                      sender().iface_attr().cap.tag.eager.max_zcopy);
}

UCS_TEST_P(test_tag, tag_rndv_zcopy_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
    test_tag_expected(static_cast<send_func>(&test_tag::tag_rndv_zcopy),
                      sender().iface_attr().cap.tag.rndv.max_zcopy);
}

UCS_TEST_P(test_tag, tag_eager_bcopy_unexpected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_eager_bcopy),
                        sender().iface_attr().cap.tag.eager.max_bcopy);
}

UCS_TEST_P(test_tag, tag_eager_zcopy_unexpected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_eager_zcopy),
                        sender().iface_attr().cap.tag.eager.max_bcopy);
}

UCS_TEST_P(test_tag, tag_rndv_zcopy_unexpected)
{
    check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_rndv_zcopy));
}

UCS_TEST_P(test_tag, tag_eager_bcopy_wrong_tag)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_tag_wrong_tag(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_P(test_tag, tag_eager_zcopy_wrong_tag)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_tag_wrong_tag(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_P(test_tag, tag_eager_short_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT);
    test_tag_mask(static_cast<send_func>(&test_tag::tag_eager_short));
}

UCS_TEST_P(test_tag, tag_eager_bcopy_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);
    test_tag_mask(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_P(test_tag, tag_eager_zcopy_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY);
    test_tag_mask(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_P(test_tag, tag_rndv_zcopy_tag_mask)
{
    check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
    test_tag_mask(static_cast<send_func>(&test_tag::tag_rndv_zcopy));
}

UCS_TEST_P(test_tag, tag_hold_uct_desc)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY | UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);

    int n = 10;
    int msg_size = ucs_min(sender().iface_attr().cap.tag.eager.max_bcopy,
                           sender().iface_attr().cap.tag.rndv.max_zcopy);
    for (int i = 0; i < n; ++i) {
        test_tag_unexpected(static_cast<send_func>(&test_tag::tag_eager_bcopy),
                            msg_size, true);

        test_tag_unexpected(static_cast<send_func>(&test_tag::tag_rndv_zcopy),
                            msg_size, true);
    }

    for (ucs::ptr_vector<void>::const_iterator iter = m_uct_descs.begin();
         iter != m_uct_descs.end(); ++iter)
    {
        uct_iface_release_desc(*iter);
    }
}


UCS_TEST_P(test_tag, tag_send_no_tag)
{
  check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);

  uct_iface_set_am_handler(receiver().iface(), 0, am_handler,
                           NULL, UCT_CB_FLAG_SYNC);
  mapped_buffer lbuf(200, SEND_SEED, sender());
  ssize_t len = uct_ep_am_bcopy(sender().ep(0), 0, mapped_buffer::pack,
                                reinterpret_cast<void*>(&lbuf), 0);
  EXPECT_EQ(lbuf.length(), static_cast<size_t>(len));
  wait_for_flag(&is_am_received);
  EXPECT_TRUE(is_am_received);
}

UCS_TEST_P(test_tag, tag_cancel_force)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);

    const size_t length = 128;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx;
    init_recv_ctx(r_ctx, &recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));
    short_progress_loop(200);
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 1));

    short_progress_loop();

    mapped_buffer sendbuf(length, SEND_SEED, sender());
    send_ctx s_ctx(&sendbuf, 1, reinterpret_cast<uint64_t>(&r_ctx));
    ASSERT_UCS_OK(tag_eager_bcopy(sender(), s_ctx));

    // Message should arrive unexpected, since tag was cancelled
    // on the receiver.
    wait_for_flag(&r_ctx.unexp);
    check_completion(r_ctx, false, SEND_SEED);
}

UCS_TEST_P(test_tag, tag_cancel_noforce)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);

    const size_t length = 128;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx;
    init_recv_ctx(r_ctx, &recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));
    short_progress_loop(200);
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 0));

    short_progress_loop();

    // Check that completed callback has been called with CANCELED status
    // (because 0 was passed as force parameter to cancel).
    EXPECT_TRUE(r_ctx.comp);
    EXPECT_EQ(r_ctx.status, UCS_ERR_CANCELED);
}

UCS_TEST_P(test_tag, tag_limit, "TM_SYNC_RATIO?=0.0")
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY);

    const size_t length = 32;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    ucs::ptr_vector<recv_ctx> rctxs;
    recv_ctx *rctx_p;
    ucs_status_t status;

    do {
        // Can use the same recv buffer, as no sends will be issued.
        rctx_p = (new recv_ctx());
        init_recv_ctx(*rctx_p, &recvbuf, 1);
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

    // Check we can post again within a reasonable time
    ucs_time_t deadline = ucs_get_time() + ucs_time_from_sec(20.0);
    do {
        status = tag_post(receiver(), rctxs.at(0));
    } while ((ucs_get_time() < deadline) && (status == UCS_ERR_EXCEEDS_LIMIT));
    ASSERT_UCS_OK(status);
}

UCS_TEST_P(test_tag, sw_rndv_expected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY | UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);

    uct_tag_t    tag    = 11;
    const size_t length = sender().iface_attr().cap.tag.rndv.max_hdr;

    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx;
    init_recv_ctx(r_ctx, &recvbuf, tag);
    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

    short_progress_loop();

    mapped_buffer sendbuf(length, SEND_SEED, sender());
    send_ctx s_ctx(&sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));

    ASSERT_UCS_OK(uct_ep_tag_rndv_request(sender().ep(0), s_ctx.tag,
                                          s_ctx.mbuf->ptr(),
                                          s_ctx.mbuf->length(), 0));

    wait_for_flag(&r_ctx.sw_rndv);

    check_completion(r_ctx, true, SEND_SEED, UCS_OK, true);

    flush();
}

UCS_TEST_P(test_tag, rndv_limit)
{
    check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);

    mapped_buffer sendbuf(8, SEND_SEED, sender());
    ucs::ptr_vector<send_ctx> sctxs;
    ucs_status_t status;
    send_ctx *sctx_p;
    void *op;

    do {
        sctx_p = new send_ctx(&sendbuf, 0xffff, 0);
        status = tag_rndv_zcopy(sender(), *sctx_p);
        sctxs.push_back(sctx_p);
    } while (status == UCS_OK);

    EXPECT_EQ(status, UCS_ERR_NO_RESOURCE);

    for (ucs::ptr_vector<send_ctx>::const_iterator iter = sctxs.begin();
         iter != sctxs.end(); ++iter)
    {
        op = (*iter)->rndv_op;
        if (!UCS_PTR_IS_ERR(op)) {
            tag_rndv_cancel(sender(), op);
        }
    }

    ucs_log_push_handler(log_ep_destroy);
    sender().destroy_eps();
    ucs_log_pop_handler();
}

UCS_TEST_P(test_tag, sw_rndv_unexpected)
{
    check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY | UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_rndv_request));
}

_UCT_INSTANTIATE_TEST_CASE(test_tag, rc)
_UCT_INSTANTIATE_TEST_CASE(test_tag, dc)
_UCT_INSTANTIATE_TEST_CASE(test_tag, rc_mlx5)
_UCT_INSTANTIATE_TEST_CASE(test_tag, dc_mlx5)
