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

#define UCT_TAG_INSTANTIATE_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_mlx5) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, dc_mlx5)

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
        mapped_buffer    *mbuf;
        void             *rndv_op;
        uct_tag_t        tag;
        uint64_t         imm_data;
        uct_completion_t uct_comp;
        ucs_status_t     status;
        bool             sw_rndv;
        bool             comp;
        bool             unexp;
    };

    typedef ucs_status_t (test_tag::*send_func)(entity&, send_ctx&);

    void init()
    {
        ucs_status_t status = uct_config_modify(m_iface_config,
                                                "RC_TM_ENABLE", "y");
        ASSERT_TRUE((status == UCS_OK) || (status == UCS_ERR_NO_ELEM));

        status = uct_config_modify(m_iface_config, "RC_TM_MP_SRQ_ENABLE", "no");
        ASSERT_TRUE((status == UCS_OK) || (status == UCS_ERR_NO_ELEM));

        uct_test::init();

        entity *sender = uct_test::create_entity(0ul, NULL, unexp_eager,
                                                 unexp_rndv,
                                                 reinterpret_cast<void*>(this),
                                                 reinterpret_cast<void*>(this));
        m_entities.push_back(sender);

        check_skip_test();

        if (UCT_DEVICE_TYPE_SELF == GetParam()->dev_type) {
            sender->connect(0, *sender, 0);
        } else {
            entity *receiver = uct_test::create_entity(0ul, NULL, unexp_eager,
                                                       unexp_rndv,
                                                       reinterpret_cast<void*>(this),
                                                       reinterpret_cast<void*>(this));
            m_entities.push_back(receiver);

            sender->connect(0, *receiver, 0);
        }
    }

    void init_send_ctx(send_ctx &s,mapped_buffer *b, uct_tag_t t, uint64_t i,
                       bool unexp_flow = true)
    {
        s.mbuf            = b;
        s.rndv_op         = NULL;
        s.tag             = t;
        s.imm_data        = i;
        s.uct_comp.count  = 1;
        s.uct_comp.status = UCS_OK;
        s.uct_comp.func   = send_completion;
        s.sw_rndv         = s.comp = false;
        s.unexp           = unexp_flow;
        s.status          = UCS_ERR_NO_PROGRESS;
    }

    void init_recv_ctx(recv_ctx &r,  mapped_buffer *b, uct_tag_t t,
                       uct_tag_t m = MASK, bool uct_d = false)
    {
        r.mbuf                    = b;
        r.tag                     = t;
        r.tmask                   = m;
        r.uct_ctx.completed_cb    = completed;
        r.uct_ctx.tag_consumed_cb = tag_consumed;
        r.uct_ctx.rndv_cb         = sw_rndv_completed;
        r.take_uct_desc           = uct_d;
        r.status                  = UCS_ERR_NO_PROGRESS;
        r.comp = r.unexp = r.consumed = r.sw_rndv = false;
    }

    ucs_status_t tag_eager_short(entity &e, send_ctx &ctx)
    {
        ctx.status = uct_ep_tag_eager_short(e.ep(0), ctx.tag, ctx.mbuf->ptr(),
                                            ctx.mbuf->length());
        ctx.comp   = true;

        return ctx.status;
    }

    ucs_status_t tag_eager_bcopy(entity &e, send_ctx &ctx)
    {
        ssize_t status = uct_ep_tag_eager_bcopy(e.ep(0), ctx.tag,
                                                ctx.imm_data, mapped_buffer::pack,
                                                reinterpret_cast<void*>(ctx.mbuf),
                                                0);
        ctx.status = (status >= 0) ? UCS_OK : static_cast<ucs_status_t>(status);
        ctx.comp   = true;

        return ctx.status;
    }

    ucs_status_t tag_eager_zcopy(entity &e, send_ctx &ctx)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, ctx.mbuf->ptr(),
                                ctx.mbuf->length(), ctx.mbuf->memh(),
                                e.iface_attr().cap.tag.eager.max_iov);

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
        ctx.sw_rndv = true;

        if (ctx.unexp) {
            // Unexpected flow, will need to analyze ctx data on the receiver
            rndv_hdr hdr = {{ctx.imm_data,
                             reinterpret_cast<uint64_t>(&ctx)
                            },
                            0xFAFA
                           };
            ctx.status = uct_ep_tag_rndv_request(e.ep(0), ctx.tag, &hdr,
                                                 sizeof(hdr), 0);
        } else {
            // Expected flow, send just plain data (will be stored in rx buf by HCA)
            ctx.status = uct_ep_tag_rndv_request(e.ep(0), ctx.tag, ctx.mbuf->ptr(),
                                                 ctx.mbuf->length(), 0);
        }
        ctx.comp = true;

        return ctx.status;
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
    void check_rx_completion(recv_ctx &ctx, bool is_expected, uint64_t seed,
                             ucs_status_t status = UCS_OK, bool is_sw_rndv = false)
    {
        EXPECT_EQ(ctx.consumed, is_expected);
        EXPECT_EQ(ctx.comp,     (is_expected && !is_sw_rndv));
        EXPECT_EQ(ctx.unexp,    (!is_expected && !is_sw_rndv));
        EXPECT_EQ(ctx.sw_rndv,  is_sw_rndv);
        EXPECT_EQ(ctx.status,   status);
        if (is_expected) {
            ctx.mbuf->pattern_check(seed);
        }
    }

    void check_tx_completion(send_ctx &ctx)
    {
        wait_for_flag(&ctx.comp);
        EXPECT_TRUE(ctx.comp);
        EXPECT_EQ(ctx.status, UCS_OK);
    }

    void test_tag_expected(send_func sfunc, size_t length = 75,
                           bool is_sw_rndv = false) {
        uct_tag_t tag = 11;

        if (RUNNING_ON_VALGRIND) {
            length = ucs_min(length, 128U);
        }

        mapped_buffer recvbuf(length, RECV_SEED, receiver());
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, tag);
        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        short_progress_loop();

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        send_ctx s_ctx;
        init_send_ctx(s_ctx, &sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx),
                      false);
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        // max rndv can be quite big, use increased timeout
        wait_for_flag(is_sw_rndv ? &r_ctx.sw_rndv : &r_ctx.comp,
                      3 * DEFAULT_TIMEOUT_SEC);

        check_rx_completion(r_ctx, true, SEND_SEED, UCS_OK, is_sw_rndv);

        // If it was RNDV send, need to wait send completion as well
        check_tx_completion(s_ctx);

        flush();
    }

    void test_tag_unexpected(send_func sfunc, size_t length = 75,
                             bool take_uct_desc = false)
    {
        uct_tag_t tag = 11;

        if (RUNNING_ON_VALGRIND) {
            length = ucs_min(length, 128U);
        }

        mapped_buffer recvbuf(length, RECV_SEED, receiver());
        mapped_buffer sendbuf(length, SEND_SEED, sender());
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, tag, MASK, take_uct_desc);
        send_ctx s_ctx;
        init_send_ctx(s_ctx, &sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        wait_for_flag(&r_ctx.unexp);
        if (static_cast<send_func>(&test_tag::tag_rndv_zcopy) == sfunc) {
            // Need to cancel origin RNDV operation, beacuse no RNDV_COMP
            // will be received (as it arrived unexpectedly and should be
            // handled by SW).
            ASSERT_UCS_OK(tag_rndv_cancel(sender(), s_ctx.rndv_op));
        }

        check_rx_completion(r_ctx, false, SEND_SEED);
        flush();
    }

    void test_tag_wrong_tag(send_func sfunc)
    {
        const size_t length = 65;
        uct_tag_t    tag    = 11;

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        mapped_buffer recvbuf(length, RECV_SEED, receiver());

        // Post modified tag for incoming message to be reported as unexpected
        // and not to be macthed.
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, tag + 1);
        send_ctx s_ctx;
        init_send_ctx(s_ctx, &sendbuf, tag, reinterpret_cast<uint64_t>(&r_ctx));

        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));

        short_progress_loop();

        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        wait_for_flag(&r_ctx.unexp);

        // Message should be reported as unexpected and filled with
        // recv seed (unchanged), as the incoming tag does not match the expected
        check_rx_completion(r_ctx, false, RECV_SEED);
        ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 1));
        flush();
    }

    void test_tag_mask(send_func sfunc)
    {
        const size_t length = 65;

        mapped_buffer recvbuf(length, RECV_SEED, receiver());

        // Post tag and tag mask in a way that it matches sender tag with
        // tag_mask applied, but is not exactly the same.
        recv_ctx r_ctx;
        init_recv_ctx(r_ctx, &recvbuf, 0xff, 0xff);
        ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

        short_progress_loop();

        mapped_buffer sendbuf(length, SEND_SEED, sender());
        send_ctx s_ctx;
        init_send_ctx(s_ctx, &sendbuf, 0xffff, reinterpret_cast<uint64_t>(&r_ctx));
        ASSERT_UCS_OK((this->*sfunc)(sender(), s_ctx));
        wait_for_flag(&r_ctx.comp);

        // Should be matched because tags are equal with tag mask applied.
        check_rx_completion(r_ctx, true, SEND_SEED);

        // If it was RNDV send, need to wait send completion as well
        check_tx_completion(s_ctx);
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

    static void verify_completed(recv_ctx *user_ctx, uct_tag_t stag, size_t length)
    {
        EXPECT_EQ(user_ctx->tag, (stag & user_ctx->tmask));
        EXPECT_EQ(user_ctx->mbuf->length(), length);
    }

    static void completed(uct_tag_context_t *self, uct_tag_t stag, uint64_t imm,
                          size_t length, void *inline_data, ucs_status_t status)
    {
        recv_ctx *user_ctx = ucs_container_of(self, recv_ctx, uct_ctx);
        user_ctx->comp     = true;
        user_ctx->status   = status;
        verify_completed(user_ctx, stag, length);
    }

    static void sw_rndv_completed(uct_tag_context_t *self, uct_tag_t stag,
                                  const void *header, unsigned header_length,
                                  ucs_status_t status, unsigned flags)
    {
        recv_ctx *user_ctx = ucs_container_of(self, recv_ctx, uct_ctx);
        user_ctx->sw_rndv  = true;
        user_ctx->status   = status;
        if (flags & UCT_TAG_RECV_CB_INLINE_DATA) {
            memcpy(user_ctx->mbuf->ptr(), header, header_length);
        }
        verify_completed(user_ctx, stag, header_length);
    }

    static ucs_status_t unexp_eager(void *arg, void *data, size_t length,
                                    unsigned flags, uct_tag_t stag,
                                    uint64_t imm, void **context)
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
                                   unsigned flags)
    {
        is_am_received = true;
        return UCS_OK;
    }

    static ucs_log_func_rc_t
    log_ep_destroy(const char *file, unsigned line, const char *function,
                   ucs_log_level_t level,
                   const ucs_log_component_config_t *comp_conf,
                   const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_WARN) {
            // Ignore warnings about uncompleted operations during ep destroy
            return UCS_LOG_FUNC_RC_STOP;
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static void send_completion(uct_completion_t *self)
    {
        send_ctx *user_ctx = ucs_container_of(self, send_ctx, uct_comp);
        user_ctx->comp     = true;
        user_ctx->status   = self->status;
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

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_short_expected,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT))
{
    test_tag_expected(static_cast<send_func>(&test_tag::tag_eager_short),
                      sender().iface_attr().cap.tag.eager.max_short);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_bcopy_expected,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    test_tag_expected(static_cast<send_func>(&test_tag::tag_eager_bcopy),
                      sender().iface_attr().cap.tag.eager.max_bcopy);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_zcopy_expected,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY))
{
    test_tag_expected(static_cast<send_func>(&test_tag::tag_eager_zcopy),
                      sender().iface_attr().cap.tag.eager.max_zcopy);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_rndv_zcopy_expected,
                     !check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY))
{
    test_tag_expected(static_cast<send_func>(&test_tag::tag_rndv_zcopy),
                      sender().iface_attr().cap.tag.rndv.max_zcopy);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_bcopy_unexpected,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_eager_bcopy),
                        sender().iface_attr().cap.tag.eager.max_bcopy);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_zcopy_unexpected,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY))
{
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_eager_zcopy),
                        sender().iface_attr().cap.tag.eager.max_bcopy);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_rndv_zcopy_unexpected,
                     !check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY))
{
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_rndv_zcopy));
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_bcopy_wrong_tag,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    test_tag_wrong_tag(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_zcopy_wrong_tag,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY))
{
    test_tag_wrong_tag(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_short_tag_mask,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT))
{
    test_tag_mask(static_cast<send_func>(&test_tag::tag_eager_short));
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_bcopy_tag_mask,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    test_tag_mask(static_cast<send_func>(&test_tag::tag_eager_bcopy));
}

UCS_TEST_SKIP_COND_P(test_tag, tag_eager_zcopy_tag_mask,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY))
{
    test_tag_mask(static_cast<send_func>(&test_tag::tag_eager_zcopy));
}

UCS_TEST_SKIP_COND_P(test_tag, tag_rndv_zcopy_tag_mask,
                     !check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY))
{
    test_tag_mask(static_cast<send_func>(&test_tag::tag_rndv_zcopy));
}

UCS_TEST_SKIP_COND_P(test_tag, tag_hold_uct_desc,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                 UCT_IFACE_FLAG_TAG_RNDV_ZCOPY))
{
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


UCS_TEST_SKIP_COND_P(test_tag, tag_send_no_tag,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    uct_iface_set_am_handler(receiver().iface(), 0, am_handler, NULL, 0);
    mapped_buffer lbuf(200, SEND_SEED, sender());
    ssize_t len = uct_ep_am_bcopy(sender().ep(0), 0, mapped_buffer::pack,
                                  reinterpret_cast<void*>(&lbuf), 0);
    EXPECT_EQ(lbuf.length(), static_cast<size_t>(len));
    wait_for_flag(&is_am_received);
    EXPECT_TRUE(is_am_received);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_cancel_force,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    const size_t length = 128;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx;
    init_recv_ctx(r_ctx, &recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));
    short_progress_loop(200);
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 1));

    short_progress_loop();

    mapped_buffer sendbuf(length, SEND_SEED, sender());
    send_ctx s_ctx;
    init_send_ctx(s_ctx, &sendbuf, 1, reinterpret_cast<uint64_t>(&r_ctx));
    ASSERT_UCS_OK(tag_eager_bcopy(sender(), s_ctx));

    // Message should arrive unexpected, since tag was cancelled
    // on the receiver.
    wait_for_flag(&r_ctx.unexp);
    check_rx_completion(r_ctx, false, SEND_SEED);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_cancel_noforce,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    const size_t length = 128;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx;
    init_recv_ctx(r_ctx, &recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));
    short_progress_loop(200);
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 0));

    wait_for_flag(&r_ctx.comp);

    // Check that completed callback has been called with CANCELED status
    // (because 0 was passed as force parameter to cancel).
    EXPECT_TRUE(r_ctx.comp);
    EXPECT_EQ(r_ctx.status, UCS_ERR_CANCELED);
}

UCS_TEST_SKIP_COND_P(test_tag, tag_limit,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    const size_t length = 32;
    ucs::ptr_vector<recv_ctx> rctxs;
    ucs::ptr_vector<mapped_buffer> rbufs;
    ucs_status_t status;

    do {
        recv_ctx *rctx_p     = new recv_ctx();
        mapped_buffer *buf_p = new mapped_buffer(length, RECV_SEED, receiver());
        init_recv_ctx(*rctx_p, buf_p, 1);
        rctxs.push_back(rctx_p);
        rbufs.push_back(buf_p);
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

    // remove posted tags from HW
    for (ucs::ptr_vector<recv_ctx>::const_iterator iter = rctxs.begin();
         iter != rctxs.end() - 1; ++iter) {
        ASSERT_UCS_OK(tag_cancel(receiver(), **iter, 1));
    }
}

UCS_TEST_SKIP_COND_P(test_tag, tag_post_same,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    const size_t length = 128;
    mapped_buffer recvbuf(length, RECV_SEED, receiver());
    recv_ctx r_ctx;
    init_recv_ctx(r_ctx, &recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

    // Can't post the same buffer until it is completed/cancelled
    ucs_status_t status = tag_post(receiver(), r_ctx);
    EXPECT_EQ(status, UCS_ERR_ALREADY_EXISTS);

    // Cancel with force, should be able to re-post immediately
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 1));
    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

    // Cancel without force, should be able to re-post when receive completion
    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 0));
    status = tag_post(receiver(), r_ctx);
    EXPECT_EQ(status, UCS_ERR_ALREADY_EXISTS); // no completion yet

    wait_for_flag(&r_ctx.comp); // cancel completed, should be able to post
    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

    // Now send something to trigger rx completion
    init_recv_ctx(r_ctx, &recvbuf, 1); // reinit rx to clear completed states
    mapped_buffer sendbuf(length, SEND_SEED, sender());
    send_ctx s_ctx;
    init_send_ctx(s_ctx, &sendbuf, 1, reinterpret_cast<uint64_t>(&r_ctx));
    ASSERT_UCS_OK(tag_eager_bcopy(sender(), s_ctx));

    wait_for_flag(&r_ctx.comp); // message consumed, should be able to post
    ASSERT_UCS_OK(tag_post(receiver(), r_ctx));

    ASSERT_UCS_OK(tag_cancel(receiver(), r_ctx, 1));
}

UCS_TEST_SKIP_COND_P(test_tag, sw_rndv_expected,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                 UCT_IFACE_FLAG_TAG_RNDV_ZCOPY))
{
    test_tag_expected(static_cast<send_func>(&test_tag::tag_rndv_request),
                      sender().iface_attr().cap.tag.rndv.max_hdr, true);
}

UCS_TEST_SKIP_COND_P(test_tag, rndv_limit,
                     !check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY))
{
    mapped_buffer sendbuf(8, SEND_SEED, sender());
    ucs::ptr_vector<send_ctx> sctxs;
    ucs_status_t status;
    send_ctx *sctx_p;
    void *op;

    do {
        sctx_p = new send_ctx;
        init_send_ctx(*sctx_p, &sendbuf, 0xffff, 0);
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

UCS_TEST_SKIP_COND_P(test_tag, sw_rndv_unexpected,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                 UCT_IFACE_FLAG_TAG_RNDV_ZCOPY))
{
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_rndv_request));
}

UCT_TAG_INSTANTIATE_TEST_CASE(test_tag)


#if defined (ENABLE_STATS) && IBV_HW_TM
extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/rc/accel/rc_mlx5_common.h>
#include <uct/ib/base/ib_verbs.h>
}

class test_tag_stats : public test_tag {
public:
    void init() {
        stats_activate();
        test_tag::init();
    }

    void cleanup() {
        test_tag::cleanup();
        stats_restore();
    }

    ucs_stats_node_t *ep_stats(const entity &e)
    {
        return ucs_derived_of(e.ep(0), uct_base_ep_t)->stats;
    }

    ucs_stats_node_t *iface_stats(const entity &e)
    {
        return ucs_derived_of(e.iface(), uct_rc_mlx5_iface_common_t)->tm.stats;
    }

    void provoke_sync(const entity &e)
    {
        uct_rc_mlx5_iface_common_t *iface;

        iface = ucs_derived_of(e.iface(), uct_rc_mlx5_iface_common_t);

        // Counters are synced every IBV_DEVICE_MAX_UNEXP_COUNT ops, set
        // it one op before, so that any following unexpected message would
        // cause HW ans SW counters sync.
        iface->tm.unexpected_cnt = IBV_DEVICE_MAX_UNEXP_COUNT - 1;
    }

    void check_tx_counters(int op, uint64_t op_val, int type, size_t len)
    {
        uint64_t v;

        v = UCS_STATS_GET_COUNTER(ep_stats(sender()), op);
        EXPECT_EQ(op_val, v);

        // With valgrind reduced messages is sent
        if (!RUNNING_ON_VALGRIND) {
            v = UCS_STATS_GET_COUNTER(ep_stats(sender()), type);
            EXPECT_EQ(len, v);
        }
    }

    void check_rx_counter(int op, uint64_t val, entity &e)
    {
        EXPECT_EQ(val, UCS_STATS_GET_COUNTER(iface_stats(e), op));
    }
};

UCS_TEST_SKIP_COND_P(test_tag_stats, tag_expected_eager,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT |
                                 UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                 UCT_IFACE_FLAG_TAG_EAGER_ZCOPY))
{
    std::pair<send_func, std::pair<size_t, int> > sfuncs[3] = {
                std::make_pair(static_cast<send_func>(&test_tag::tag_eager_short),
                std::make_pair(sender().iface_attr().cap.tag.eager.max_short,
                static_cast<int>(UCT_EP_STAT_BYTES_SHORT))),

                std::make_pair(static_cast<send_func>(&test_tag::tag_eager_bcopy),
                std::make_pair(sender().iface_attr().cap.tag.eager.max_bcopy,
                static_cast<int>(UCT_EP_STAT_BYTES_BCOPY))),

                std::make_pair(static_cast<send_func>(&test_tag::tag_eager_zcopy),
                std::make_pair(sender().iface_attr().cap.tag.eager.max_zcopy,
                static_cast<int>(UCT_EP_STAT_BYTES_ZCOPY)))
    };

    for (int i = 0; i < 3; ++i) {
        test_tag_expected(sfuncs[i].first, sfuncs[i].second.first);
        check_tx_counters(UCT_EP_STAT_TAG, i + 1,
                          sfuncs[i].second.second,
                          sfuncs[i].second.first);
        check_rx_counter(UCT_RC_MLX5_STAT_TAG_RX_EXP, i + 1, receiver());
    }
}

UCS_TEST_SKIP_COND_P(test_tag_stats, tag_unexpected_eager,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                 UCT_IFACE_FLAG_TAG_EAGER_ZCOPY))
{
    std::pair<send_func, std::pair<size_t, int> > sfuncs[2] = {
                std::make_pair(static_cast<send_func>(&test_tag::tag_eager_bcopy),
                std::make_pair(sender().iface_attr().cap.tag.eager.max_bcopy,
                static_cast<int>(UCT_EP_STAT_BYTES_BCOPY))),

                std::make_pair(static_cast<send_func>(&test_tag::tag_eager_zcopy),
                std::make_pair(sender().iface_attr().cap.tag.eager.max_zcopy,
                static_cast<int>(UCT_EP_STAT_BYTES_ZCOPY)))
    };

    for (int i = 0; i < 2; ++i) {
        test_tag_unexpected(sfuncs[i].first, sfuncs[i].second.first);
        check_tx_counters(UCT_EP_STAT_TAG, i + 1,
                          sfuncs[i].second.second,
                          sfuncs[i].second.first);
        check_rx_counter(UCT_RC_MLX5_STAT_TAG_RX_EAGER_UNEXP, i + 1, receiver());
    }
}

UCS_TEST_SKIP_COND_P(test_tag_stats, tag_list_ops,
                     !check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    mapped_buffer recvbuf(32, RECV_SEED, receiver());
    recv_ctx rctx;

    init_recv_ctx(rctx, &recvbuf, 1);

    ASSERT_UCS_OK(tag_post(receiver(), rctx));
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_LIST_ADD, 1ul, receiver());

    ASSERT_UCS_OK(tag_cancel(receiver(), rctx, 1));
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_LIST_DEL, 1ul, receiver());

    // Every ADD and DEL is paired with SYNC, but stats counter is increased
    // when separate SYNC op is issued only. So, we expect it to be 0 after
    // ADD and DEL operations.
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_LIST_SYNC, 0ul, receiver());

    // Provoke real SYNC op and send a message unexpectedly
    provoke_sync(receiver());
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_eager_bcopy));
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_LIST_SYNC, 1ul, receiver());
}


UCS_TEST_SKIP_COND_P(test_tag_stats, tag_rndv,
                     !check_caps(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY |
                                 UCT_IFACE_FLAG_TAG_EAGER_BCOPY))
{
    size_t len = sender().iface_attr().cap.tag.rndv.max_zcopy / 8;

    // Check UNEXP_RNDV on the receiver
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_rndv_zcopy), len);
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_RX_RNDV_UNEXP, 1ul, receiver());

    // Check that sender receives RNDV_FIN in case of expected rndv message
    test_tag_expected(static_cast<send_func>(&test_tag::tag_rndv_zcopy), len);
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_RX_RNDV_FIN, 1ul, sender());


    // Check UNEXP_RNDV_REQ on the receiver
    test_tag_unexpected(static_cast<send_func>(&test_tag::tag_rndv_request));
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_RX_RNDV_REQ_UNEXP, 1ul, receiver());

    // Check NEXP_RNDV_REQ on the receiver
    test_tag_expected(static_cast<send_func>(&test_tag::tag_rndv_request),
                     sender().iface_attr().cap.tag.rndv.max_hdr, true);
    check_rx_counter(UCT_RC_MLX5_STAT_TAG_RX_RNDV_REQ_EXP, 1ul, receiver());
}

UCT_TAG_INSTANTIATE_TEST_CASE(test_tag_stats)

#endif


#if IBV_HW_TM

extern "C" {
#include <uct/ib/rc/accel/rc_mlx5_common.h>
}

// TODO: Unite with test_tag + add GRH testing for DC
class test_tag_mp_xrq : public uct_test {
public:
    static const uint64_t SEND_SEED = 0xa1a1a1a1a1a1a1a1ul;
    static const uint64_t AM_ID     = 1;
    typedef void (test_tag_mp_xrq::*send_func)(mapped_buffer*);

    virtual void init();
    test_tag_mp_xrq();
    uct_rc_mlx5_iface_common_t* rc_mlx5_iface(entity &e);
    void send_eager_bcopy(mapped_buffer *buf);
    void send_eager_zcopy(mapped_buffer *buf);
    void send_rndv_zcopy(mapped_buffer *buf);
    void send_rndv_request(mapped_buffer *buf);
    void send_am_bcopy(mapped_buffer *buf);
    void test_common(send_func sfunc, size_t num_segs, size_t exp_segs = 1,
                     bool is_eager = true);

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags);

    static ucs_status_t unexp_eager(void *arg, void *data, size_t length,
                                    unsigned flags, uct_tag_t stag,
                                    uint64_t imm, void **context);

    static ucs_status_t unexp_rndv(void *arg, unsigned flags, uint64_t stag,
                                   const void *header, unsigned header_length,
                                   uint64_t remote_addr, size_t length,
                                   const void *rkey_buf);

protected:
    static size_t      m_rx_counter;
    std::vector<void*> m_uct_descs;
    bool               m_hold_uct_desc;

    uct_test::entity& sender() {
        return **m_entities.begin();
    }

    uct_test::entity& receiver() {
        return **(m_entities.end() - 1);
    }

private:
    ucs_status_t unexp_handler(void *data, unsigned flags, uint64_t imm,
                               void **context);
    ucs_status_t handle_uct_desc(void *data, unsigned flags);
    void set_env_var_or_skip(void *config, const char *var, const char *val);
    size_t           m_max_hdr;
    bool             m_first_received;
    bool             m_last_received;
    uct_completion_t m_uct_comp;
};

size_t test_tag_mp_xrq::m_rx_counter = 0;

test_tag_mp_xrq::test_tag_mp_xrq() : m_hold_uct_desc(false),
                                     m_first_received(false),
                                     m_last_received(false)
{
    m_max_hdr         = sizeof(ibv_tmh) + sizeof(ibv_rvh);
    m_uct_comp.count  = 512; // We do not need completion func to be invoked
    m_uct_comp.status = UCS_OK;
    m_uct_comp.func   = NULL;
}

uct_rc_mlx5_iface_common_t* test_tag_mp_xrq::rc_mlx5_iface(entity &e)
{
    return ucs_derived_of(e.iface(), uct_rc_mlx5_iface_common_t);
}

void test_tag_mp_xrq::set_env_var_or_skip(void *config, const char *var,
                                          const char *val)
{
    ucs_status_t status = uct_config_modify(config, var, val);
    if (status != UCS_OK) {
        ucs_warn("%s", ucs_status_string(status));
        UCS_TEST_SKIP_R(std::string("Can't set ") + var);
    }
}

void test_tag_mp_xrq::init()
{
    set_env_var_or_skip(m_iface_config, "RC_TM_ENABLE", "y");
    set_env_var_or_skip(m_iface_config, "RC_TM_MP_SRQ_ENABLE", "try");
    set_env_var_or_skip(m_iface_config, "RC_TM_MP_NUM_STRIDES", "8");
    set_env_var_or_skip(m_md_config, "MLX5_DEVX_OBJECTS", "dct,dcsrq,rcsrq,rcqp");

    uct_test::init();

    entity *sender = uct_test::create_entity(0ul, NULL, unexp_eager, unexp_rndv,
                                             reinterpret_cast<void*>(this),
                                             reinterpret_cast<void*>(this));
    m_entities.push_back(sender);

    entity *receiver = uct_test::create_entity(0ul, NULL, unexp_eager, unexp_rndv,
                                               reinterpret_cast<void*>(this),
                                               reinterpret_cast<void*>(this));
    m_entities.push_back(receiver);

    if (!UCT_RC_MLX5_MP_ENABLED(rc_mlx5_iface(test_tag_mp_xrq::sender()))) {
        UCS_TEST_SKIP_R("No MP XRQ support");
    }

    sender->connect(0, *receiver, 0);

    uct_iface_set_am_handler(receiver->iface(), AM_ID, am_handler, this, 0);
}

void test_tag_mp_xrq::send_eager_bcopy(mapped_buffer *buf)
{
    ssize_t len = uct_ep_tag_eager_bcopy(sender().ep(0), 0x11,
                                         reinterpret_cast<uint64_t>(this),
                                         mapped_buffer::pack,
                                         reinterpret_cast<void*>(buf), 0);

    EXPECT_EQ(buf->length(), static_cast<size_t>(len));
}

void test_tag_mp_xrq::send_eager_zcopy(mapped_buffer *buf)
{
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buf->ptr(), buf->length(), buf->memh(),
                            sender().iface_attr().cap.tag.eager.max_iov);

    ucs_status_t status = uct_ep_tag_eager_zcopy(sender().ep(0), 0x11,
                                                 reinterpret_cast<uint64_t>(this),
                                                 iov, iovcnt, 0, &m_uct_comp);
    ASSERT_UCS_OK_OR_INPROGRESS(status);
}

void test_tag_mp_xrq::send_rndv_zcopy(mapped_buffer *buf)
{
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buf->ptr(), buf->length(), buf->memh(),
                            sender().iface_attr().cap.tag.rndv.max_iov);

    uint64_t dummy_hdr       = 0xFAFA;
    ucs_status_ptr_t rndv_op = uct_ep_tag_rndv_zcopy(sender().ep(0), 0x11, &dummy_hdr,
                                                     sizeof(dummy_hdr), iov,
                                                     iovcnt, 0, &m_uct_comp);
    ASSERT_FALSE(UCS_PTR_IS_ERR(rndv_op));

    // There will be no real RNDV performed, cancel the op to avoid mpool
    // warning on exit
    ASSERT_UCS_OK(uct_ep_tag_rndv_cancel(sender().ep(0),rndv_op));
}

void test_tag_mp_xrq::send_rndv_request(mapped_buffer *buf)
{
    size_t size = sender().iface_attr().cap.tag.rndv.max_hdr;
    void *hdr   = alloca(size);

    ASSERT_UCS_OK(uct_ep_tag_rndv_request(sender().ep(0), 0x11, hdr, size, 0));
}

void test_tag_mp_xrq::send_am_bcopy(mapped_buffer *buf)
{
    ssize_t len = uct_ep_am_bcopy(sender().ep(0), AM_ID, mapped_buffer::pack,
                                  reinterpret_cast<void*>(buf), 0);

    EXPECT_EQ(buf->length(), static_cast<size_t>(len));
}

void test_tag_mp_xrq::test_common(send_func sfunc, size_t num_segs,
                                  size_t exp_segs, bool is_eager)
{
    size_t seg_size  = rc_mlx5_iface(sender())->super.super.config.seg_size;
    size_t seg_num   = is_eager ? num_segs : 1;
    size_t exp_val   = is_eager ? exp_segs : 1;
    size_t size      = (seg_size * seg_num) - m_max_hdr;
    m_rx_counter     = 0;
    m_first_received = m_last_received = false;

    EXPECT_TRUE(size <= sender().iface_attr().cap.tag.eager.max_bcopy);
    mapped_buffer buf(size, SEND_SEED, sender());

    (this->*sfunc)(&buf);

    wait_for_value(&m_rx_counter, exp_val, true);
    EXPECT_EQ(exp_val, m_rx_counter);
    EXPECT_EQ(is_eager, m_first_received); // relevant for eager only
    EXPECT_EQ(is_eager, m_last_received);  // relevant for eager only
}

ucs_status_t test_tag_mp_xrq::handle_uct_desc(void *data, unsigned flags)
{
    if ((flags & UCT_CB_PARAM_FLAG_DESC) && m_hold_uct_desc) {
        m_uct_descs.push_back(data);
        return UCS_INPROGRESS;
    }

    return UCS_OK;
}

ucs_status_t test_tag_mp_xrq::am_handler(void *arg, void *data, size_t length,
                                         unsigned flags)
{
   EXPECT_TRUE(flags & UCT_CB_PARAM_FLAG_FIRST);
   EXPECT_FALSE(flags & UCT_CB_PARAM_FLAG_MORE);

   m_rx_counter++;

   test_tag_mp_xrq *self = reinterpret_cast<test_tag_mp_xrq*>(arg);
   return self->handle_uct_desc(data, flags);
}

ucs_status_t test_tag_mp_xrq::unexp_handler(void *data, unsigned flags,
                                            uint64_t imm, void **context)
{
    void *self = reinterpret_cast<void*>(this);

    if (flags & UCT_CB_PARAM_FLAG_FIRST) {
        // Set the message context which will be passed back with the rest of
        // message fragments
        *context         = self;
        m_first_received = true;

    } else {
        // Check that the correct message context is passed with all fragments
        EXPECT_EQ(self, *context);
    }

    if (!(flags & UCT_CB_PARAM_FLAG_MORE)) {
        // Last message should contain valid immediate value
        EXPECT_EQ(reinterpret_cast<uint64_t>(this), imm);
        m_last_received = true;
    } else {
        // Immediate value is passed with the last message only
        EXPECT_EQ(0ul, imm);
    }


    return handle_uct_desc(data, flags);
}

ucs_status_t test_tag_mp_xrq::unexp_eager(void *arg, void *data, size_t length,
                                          unsigned flags, uct_tag_t stag,
                                          uint64_t imm, void **context)
{
    test_tag_mp_xrq *self = reinterpret_cast<test_tag_mp_xrq*>(arg);

    m_rx_counter++;

    return self->unexp_handler(data, flags, imm, context);
}

ucs_status_t test_tag_mp_xrq::unexp_rndv(void *arg, unsigned flags,
                                         uint64_t stag, const void *header,
                                         unsigned header_length,
                                         uint64_t remote_addr, size_t length,
                                         const void *rkey_buf)
{
    EXPECT_FALSE(flags & UCT_CB_PARAM_FLAG_FIRST);
    EXPECT_FALSE(flags & UCT_CB_PARAM_FLAG_MORE);

    m_rx_counter++;

    return UCS_OK;
}

UCS_TEST_P(test_tag_mp_xrq, config)
{
    uct_rc_mlx5_iface_common_t *iface = rc_mlx5_iface(sender());

    // MP XRQ is supported with tag offload only
    EXPECT_TRUE(UCT_RC_MLX5_TM_ENABLED(iface));

    // With MP XRQ segment size should be equal to MTU, because HW generates
    // CQE per each received MTU
    size_t mtu = uct_ib_mtu_value(uct_ib_iface_port_attr(&(iface)->super.super)->active_mtu);
    EXPECT_EQ(mtu, iface->super.super.config.seg_size);

    const uct_iface_attr *attrs = &sender().iface_attr();

    // Max tag bcopy is limited by tag tx memory pool
    EXPECT_EQ(iface->tm.max_bcopy - sizeof(ibv_tmh),
              attrs->cap.tag.eager.max_bcopy);
    EXPECT_GT(attrs->cap.tag.eager.max_bcopy,
              iface->super.super.config.seg_size);

    // Max tag zcopy is limited by maximal IB message size
    EXPECT_EQ(uct_ib_iface_port_attr(&iface->super.super)->max_msg_sz - sizeof(ibv_tmh),
              attrs->cap.tag.eager.max_zcopy);

    // Maximal AM size should not exceed segment size, so it would always
    // arrive in one-fragment packet (with header it should be strictly less)
    EXPECT_LT(attrs->cap.am.max_bcopy, iface->super.super.config.seg_size);
    EXPECT_LT(attrs->cap.am.max_zcopy, iface->super.super.config.seg_size);
}

UCS_TEST_P(test_tag_mp_xrq, desc_release)
{
    m_hold_uct_desc = true; // We want to "hold" UCT memory descriptors
    std::pair<send_func, bool> sfuncs[5] = {
              std::make_pair(&test_tag_mp_xrq::send_eager_bcopy,  true),
              std::make_pair(&test_tag_mp_xrq::send_eager_zcopy,  true),
              std::make_pair(&test_tag_mp_xrq::send_rndv_zcopy,   false),
              std::make_pair(&test_tag_mp_xrq::send_rndv_request, false),
              std::make_pair(&test_tag_mp_xrq::send_am_bcopy,     false)
    };

    for (int i = 0; i < 5; ++i) {
        test_common(sfuncs[i].first, 3, 3, sfuncs[i].second);
    }

    for (ucs::ptr_vector<void>::const_iterator iter = m_uct_descs.begin();
         iter != m_uct_descs.end(); ++iter)
    {
        uct_iface_release_desc(*iter);
    }
}

UCS_TEST_P(test_tag_mp_xrq, am)
{
    test_common(&test_tag_mp_xrq::send_am_bcopy, 1, 1, false);
}

UCS_TEST_P(test_tag_mp_xrq, bcopy_eager_only)
{
    test_common(&test_tag_mp_xrq::send_eager_bcopy, 1);
}

UCS_TEST_P(test_tag_mp_xrq, zcopy_eager_only)
{
    test_common(&test_tag_mp_xrq::send_eager_zcopy, 1);
}

UCS_TEST_P(test_tag_mp_xrq, bcopy_eager)
{
    test_common(&test_tag_mp_xrq::send_eager_bcopy, 5, 5);
}

UCS_TEST_P(test_tag_mp_xrq, zcopy_eager)
{
    test_common(&test_tag_mp_xrq::send_eager_zcopy, 5, 5);
}

UCS_TEST_P(test_tag_mp_xrq, rndv_zcopy)
{
    test_common(&test_tag_mp_xrq::send_rndv_zcopy, 1, 1, false);
}

UCS_TEST_P(test_tag_mp_xrq, rndv_request)
{
    test_common(&test_tag_mp_xrq::send_rndv_request, 1, 1, false);
}

UCT_TAG_INSTANTIATE_TEST_CASE(test_tag_mp_xrq)

#endif
