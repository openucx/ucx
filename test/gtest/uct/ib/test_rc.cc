/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

#include "test_rc.h"
#include <uct/ib/rc/verbs/rc_verbs.h>
#include <uct/test_peer_failure.h>

#ifdef HAVE_MLX5_DV
extern "C" {
#include <uct/ib/mlx5/rc/rc_mlx5_common.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>
}

#include <sched.h>
#endif


void test_rc::init()
{
    uct_test::init();

    m_e1 = uct_test::create_entity(0);
    m_entities.push_back(m_e1);

    check_skip_test();

    m_e2 = uct_test::create_entity(0);
    m_entities.push_back(m_e2);

    connect();
}

void test_rc::connect()
{
    m_e1->connect(0, *m_e2, 0);
    m_e2->connect(0, *m_e1, 0);

    uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler, NULL, 0);
    uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler, NULL, 0);
}

// Check that iface tx ops buffer and flush comp memory pool are moderated
// properly when we have communication ops + lots of flushes
void test_rc::test_iface_ops(int cq_len)
{
    entity *e = uct_test::create_entity(0);
    m_entities.push_back(e);
    e->connect(0, *m_e2, 0);

    mapped_buffer sendbuf(1024, 0ul, *e);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);
    uct_completion_t comp;
    comp.count = cq_len * 512; // some big value to avoid func invocation
    comp.func  = NULL;

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(), m_e1->iface_attr().cap.put.max_iov);
    // For _x transports several CQEs can be consumed per WQE, post less put zcopy
    // ops, so that flush would be successful (otherwise flush will return
    // NO_RESOURCES and completion will not be added for it).
    for (int i = 0; i < cq_len / 5; i++) {
        ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_put_zcopy(e->ep(0), iov, iovcnt,
                                                     recvbuf.addr(),
                                                     recvbuf.rkey(), &comp));

        // Create some stress on iface (flush mp):
        // post 10 flushes per every put.
        for (int j = 0; j < 10; j++) {
            ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_flush(e->ep(0), 0, &comp));
        }
    }

    flush();
}

UCS_TEST_SKIP_COND_P(test_rc, stress_iface_ops,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY)) {
    int cq_len = 16;

    if (UCS_OK != uct_config_modify(m_iface_config, "RC_TX_CQ_LEN",
                                    ucs::to_string(cq_len).c_str())) {
        UCS_TEST_ABORT("Error: cannot modify RC_TX_CQ_LEN");
    }

    test_iface_ops(cq_len);
}

UCS_TEST_P(test_rc, tx_cq_moderation) {
    unsigned tx_mod   = ucs_min(rc_iface(m_e1)->config.tx_moderation / 4, 8);
    int16_t init_rsc  = rc_ep(m_e1)->txqp.available;

    send_am_messages(m_e1, tx_mod, UCS_OK);

    int16_t rsc = rc_ep(m_e1)->txqp.available;

    EXPECT_LE(rsc, init_rsc);

    short_progress_loop(100);

    EXPECT_EQ(rsc, rc_ep(m_e1)->txqp.available);

    flush();

    EXPECT_EQ(init_rsc, rc_ep(m_e1)->txqp.available);
}

UCS_TEST_P(test_rc, flush_fc, "FLUSH_MODE?=fc") {
    send_am_messages(m_e1, 1, UCS_OK);

    ucs_status_t status;
    do {
        status = uct_ep_flush(m_e1->ep(0), 0, NULL);
        short_progress_loop();
        if (status != UCS_ERR_NO_RESOURCE) {
            ASSERT_UCS_OK_OR_INPROGRESS(status);
        }
    } while (status != UCS_OK);
}

UCS_TEST_P(test_rc, fence_am_short_consumed, "RC_FENCE=weak")
{
    uct_ib_fence_info_t *fence_info;

    if (GetParam()->tl_name == "rc_verbs") {
        fence_info = &ucs_derived_of(m_e1->ep(0), uct_rc_verbs_ep_t)->fi;
    } else {
#ifdef HAVE_MLX5_DV
        fence_info =
                &ucs_derived_of(m_e1->ep(0),
                                uct_rc_mlx5_ep_t)->super.tx.wq.fi;
#else
        UCS_TEST_ABORT("rc_mlx5 transport requires mlx5 DV support");
#endif
    }

    ASSERT_UCS_OK(uct_ep_fence(m_e1->ep(0), 0));
    EXPECT_NE(rc_iface(m_e1)->tx.fi.fence_beat, fence_info->fence_beat);

    ASSERT_UCS_OK(uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0));
    EXPECT_EQ(rc_iface(m_e1)->tx.fi.fence_beat, fence_info->fence_beat);
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc)

#ifdef HAVE_MLX5_DV

class test_rc_mlx5_invalidate : public test_rc {
protected:
    struct tracked_completion {
        uct_completion_t uct;
        unsigned         callback_count;
    };

    static void completion_cb(uct_completion_t *comp)
    {
        tracked_completion *tracked = ucs_container_of(comp,
                                                       tracked_completion, uct);

        ++tracked->callback_count;
    }
};

UCS_TEST_P(test_rc_mlx5_invalidate, no_completions)
{
    uct_rc_mlx5_base_ep_t *ep =
            reinterpret_cast<uct_rc_mlx5_base_ep_t*>(m_e1->ep(0));
    uct_ep_invalidate_params_t params = {};

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;

    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), &params));
    EXPECT_TRUE(ep->flags & UCT_RC_MLX5_EP_FLAG_NO_COMPLETIONS);

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  uct_ep_invalidate(m_e1->ep(0), &params));
    }
}

UCS_TEST_SKIP_COND_P(test_rc_mlx5_invalidate,
                     destroy_preserves_completion_ownership,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    const size_t length = 32;
    mapped_buffer sendbuf(length, 0ul, *m_e1);
    mapped_buffer recvbuf(length, 0ul, *m_e2);
    tracked_completion completion     = {};
    uct_ep_invalidate_params_t params = {};

    completion.uct.func  = completion_cb;
    completion.uct.count = 1;

    uct_iface_progress_disable(m_e1->iface(), UCT_PROGRESS_SEND);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(),
                            m_e1->iface_attr().cap.put.max_iov);
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(m_e1->ep(0), iov, iovcnt, recvbuf.addr(),
                               recvbuf.rkey(), &completion.uct));

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;
    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), &params));

    m_e1->destroy_ep(0);

    EXPECT_EQ(1, completion.uct.count);
    EXPECT_EQ(0u, completion.callback_count);
    uct_iface_progress_enable(m_e1->iface(), UCT_PROGRESS_SEND);
}

UCS_TEST_P(test_rc_mlx5_invalidate,
           no_completions_unsupported_with_hw_tag_matching)
{
    uct_rc_mlx5_iface_common_t *iface =
            reinterpret_cast<uct_rc_mlx5_iface_common_t*>(m_e1->iface());
    uct_rc_mlx5_base_ep_t *ep =
            reinterpret_cast<uct_rc_mlx5_base_ep_t*>(m_e1->ep(0));
    uct_ep_invalidate_params_t params = {};
    uint8_t initial_tm_enabled        = iface->tm.enabled;
    uint8_t initial_ep_flags          = ep->flags;
    ucs_status_t status;

    iface->tm.enabled = 1;

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;
    status            = uct_ep_invalidate(m_e1->ep(0), &params);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, status);
    EXPECT_EQ(initial_ep_flags, ep->flags);

    iface->tm.enabled = initial_tm_enabled;
    if (status != UCS_ERR_UNSUPPORTED) {
        scoped_log_handler hide_warn(hide_warns_logger);
        m_e1->destroy_ep(0);
        return;
    }

    ASSERT_UCS_OK(send_am_message(m_e1));
    flush();
}

UCS_TEST_P(test_rc_mlx5_invalidate, unknown_flags_is_atomic)
{
    uct_ep_invalidate_params_t params = {};
    ucs_status_t status;

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCS_BIT(31);

    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        status = uct_ep_invalidate(m_e1->ep(0), &params);
    }
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    if (status != UCS_ERR_INVALID_PARAM) {
        scoped_log_handler hide_warn(hide_warns_logger);
        m_e1->destroy_ep(0);
        return;
    }

    ASSERT_UCS_OK(send_am_message(m_e1));
    flush();
}

UCS_TEST_P(test_rc_mlx5_invalidate, null_params)
{
    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), NULL));
}

_UCT_INSTANTIATE_TEST_CASE(test_rc_mlx5_invalidate, rc_mlx5)


class test_rc_mlx5_late_cqe : public test_rc {
public:
    void init() override
    {
        modify_config("IB_TX_CQE_ZIP_ENABLE", "no");
        modify_config("IB_TX_INLINE_RESP", "64");
        test_rc::init();
    }

protected:
    struct tracked_completion {
        uct_completion_t uct;
        unsigned         callback_count;
    };

    struct error_context {
        uct_rc_mlx5_base_ep_t *ep;
        tracked_completion    *completion;
        uint16_t              observed_hw_ci;
        int16_t               observed_available;
        unsigned              observed_outstanding;
        unsigned              observed_completion_count;
        unsigned              callback_count;
    };

    class ep_cleanup_guard {
    public:
        ep_cleanup_guard(entity *test_entity) : m_entity(test_entity)
        {
        }

        ~ep_cleanup_guard()
        {
            uct_rc_mlx5_iface_common_t *iface =
                    reinterpret_cast<uct_rc_mlx5_iface_common_t*>(
                            m_entity->iface());
            uct_rc_mlx5_base_ep_t *ep =
                    reinterpret_cast<uct_rc_mlx5_base_ep_t*>(m_entity->ep(0));
            uct_ib_mlx5_txwq_t *txwq = &ep->tx.wq;
            uct_rc_txqp_t *txqp      = &ep->super.txqp;

            /* This fixture validates CQE processing, not EP retirement. */
            if (!ucs_queue_is_empty(&txqp->outstanding)) {
                uct_rc_txqp_purge_outstanding(&iface->super, txqp,
                                              UCS_ERR_CANCELED, txwq->sw_pi, 0,
                                              1);
            }

            uct_iface_progress_enable(m_entity->iface(), UCT_PROGRESS_SEND);
            scoped_log_handler hide_warn(hide_warns_logger);
            m_entity->destroy_ep(0);
        }

    private:
        entity *m_entity;
    };

    static void completion_cb(uct_completion_t *comp)
    {
        tracked_completion *tracked = ucs_container_of(comp, tracked_completion,
                                                       uct);

        ++tracked->callback_count;
    }

    static struct mlx5_cqe64 make_error_cqe(uint32_t qpn, uint16_t pi)
    {
        uct_ib_mlx5_err_cqe_t err_cqe = {};
        struct mlx5_cqe64 cqe;

        UCS_STATIC_ASSERT(sizeof(cqe) == sizeof(err_cqe));

        err_cqe.s_wqe_opcode_qpn = htonl((MLX5_OPCODE_RDMA_READ << 24) | qpn);
        err_cqe.wqe_counter      = htons(pi);
        err_cqe.syndrome         = MLX5_CQE_SYNDROME_WR_FLUSH_ERR;
        err_cqe.op_own           = MLX5_CQE_REQ_ERR << 4;
        memcpy(&cqe, &err_cqe, sizeof(cqe));
        return cqe;
    }

    static ucs_status_t
    observe_error_cb(void *arg, uct_ep_h ep, ucs_status_t status)
    {
        error_context *context = static_cast<error_context*>(arg);

        EXPECT_EQ(&context->ep->super.super.super, ep);
        EXPECT_EQ(UCS_ERR_CANCELED, status);
        context->observed_hw_ci       = context->ep->tx.wq.hw_ci;
        context->observed_available   = uct_rc_txqp_available(
                &context->ep->super.txqp);
        context->observed_outstanding = ucs_queue_length(
                &context->ep->super.txqp.outstanding);
        if (context->completion != NULL) {
            context->observed_completion_count =
                    context->completion->callback_count;
        }

        ++context->callback_count;
        return UCS_OK;
    }

    static uint16_t logical_ci(uct_rc_mlx5_base_ep_t *ep)
    {
        uct_ib_mlx5_txwq_t *txwq = &ep->tx.wq;

        return txwq->prev_sw_pi -
               (txwq->bb_max - uct_rc_txqp_available(&ep->super.txqp));
    }

    static struct mlx5_cqe64 *get_cqe(uct_ib_mlx5_cq_t *cq, unsigned cqe_index)
    {
        return reinterpret_cast<struct mlx5_cqe64*>(
                static_cast<char*>(cq->cq_buf) +
                ((cqe_index & cq->cq_length_mask) << cq->cqe_size_log));
    }

    static int cqe_is_hw_owned(uct_ib_mlx5_cq_t *cq, struct mlx5_cqe64 *cqe,
                               unsigned cqe_index)
    {
        uint8_t sw_it_count = cqe_index >> cq->cq_length_log;

        return (sw_it_count ^ cqe->op_own) & MLX5_CQE_OWNER_MASK;
    }

    static int cqe_is_error_or_zipped(uint8_t op_own)
    {
        const uint8_t mask = UCT_IB_MLX5_CQE_FORMAT_MASK |
                             UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK;

        return (op_own & mask) >= UCT_IB_MLX5_CQE_FORMAT_MASK;
    }

    static struct mlx5_cqe64 *
    wait_for_cqe(uct_ib_mlx5_cq_t *cq, unsigned cqe_index)
    {
        struct mlx5_cqe64 *cqe = get_cqe(cq, cqe_index);
        ucs_time_t deadline    = ucs::get_deadline(10.0);

        while (cqe_is_hw_owned(cq, cqe, cqe_index) &&
               (ucs_get_time() < deadline)) {
            sched_yield();
        }

        return cqe;
    }
};

UCS_TEST_SKIP_COND_P(test_rc_mlx5_late_cqe, success_after_no_completions,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    uct_rc_mlx5_iface_common_t *iface =
            reinterpret_cast<uct_rc_mlx5_iface_common_t*>(m_e1->iface());
    uct_rc_mlx5_base_ep_t *ep = reinterpret_cast<uct_rc_mlx5_base_ep_t*>(
            m_e1->ep(0));
    uct_ib_mlx5_txwq_t *txwq  = &ep->tx.wq;
    uct_rc_txqp_t *txqp       = &ep->super.txqp;
    uct_ib_mlx5_cq_t *cq      = &iface->cq[UCT_IB_DIR_TX];
    mapped_buffer sendbuf(64, 0ul, *m_e1);
    mapped_buffer recvbuf(64, 0ul, *m_e2);
    tracked_completion completion     = {};
    uct_ep_invalidate_params_t params = {};
    struct mlx5_cqe64 *cqe;
    unsigned old_cq_ci;
    uint16_t old_hw_ci, old_logical_ci, cqe_pi;
    int16_t post_available;
    signed post_cq_available;

    flush();
    memset(sendbuf.ptr(), 0xa5, sendbuf.length());
    memset(recvbuf.ptr(), 0, recvbuf.length());

    completion.uct.func   = completion_cb;
    completion.uct.count  = 1;
    completion.uct.status = UCS_OK;

    uct_iface_progress_disable(m_e1->iface(), UCT_PROGRESS_SEND);
    ep_cleanup_guard cleanup(m_e1);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(), m_e1->iface_attr().cap.put.max_iov);
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(m_e1->ep(0), iov, iovcnt, recvbuf.addr(),
                               recvbuf.rkey(), &completion.uct));

    old_cq_ci = cq->cq_ci;
    cqe       = wait_for_cqe(cq, old_cq_ci);
    ASSERT_FALSE(cqe_is_hw_owned(cq, cqe, old_cq_ci));
    ucs_memory_cpu_load_fence();
    ASSERT_FALSE(cqe_is_error_or_zipped(cqe->op_own));

    cqe_pi            = ntohs(cqe->wqe_counter);
    old_hw_ci         = txwq->hw_ci;
    old_logical_ci    = logical_ci(ep);
    post_available    = txqp->available;
    post_cq_available = iface->super.tx.cq_available;

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;
    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), &params));
    ASSERT_EQ(old_logical_ci, txwq->ft_ci);

    EXPECT_GT(uct_iface_progress(m_e1->iface()), 0u);
    EXPECT_EQ(old_cq_ci + 1, cq->cq_ci);
    EXPECT_EQ(cqe_pi, txwq->hw_ci);
    EXPECT_EQ(old_logical_ci, logical_ci(ep));
    EXPECT_EQ(old_logical_ci, txwq->ft_ci);
    EXPECT_EQ(post_available, txqp->available);
    EXPECT_EQ(post_cq_available + (uint16_t)(cqe_pi - old_hw_ci),
              iface->super.tx.cq_available);
    EXPECT_EQ(txwq->prev_sw_pi, txwq->hw_ci);
    EXPECT_LT(txqp->available, txwq->bb_max);
    EXPECT_TRUE(ucs_queue_is_empty(&txqp->outstanding));
    EXPECT_EQ(1, completion.uct.count);
    EXPECT_EQ(0u, completion.callback_count);
    EXPECT_EQ(0, memcmp(sendbuf.ptr(), recvbuf.ptr(), sendbuf.length()));
}

UCS_TEST_SKIP_COND_P(test_rc_mlx5_late_cqe, get_bcopy_after_no_completions,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY))
{
    uct_rc_mlx5_iface_common_t *iface =
            reinterpret_cast<uct_rc_mlx5_iface_common_t*>(m_e1->iface());
    uct_rc_mlx5_base_ep_t *ep = reinterpret_cast<uct_rc_mlx5_base_ep_t*>(
            m_e1->ep(0));
    uct_ib_mlx5_txwq_t *txwq  = &ep->tx.wq;
    uct_rc_txqp_t *txqp       = &ep->super.txqp;
    uct_ib_mlx5_cq_t *cq      = &iface->cq[UCT_IB_DIR_TX];
    const size_t length       = 32;
    mapped_buffer localbuf(length, 0ul, *m_e1);
    mapped_buffer remotebuf(length, 0ul, *m_e2);
    tracked_completion completion     = {};
    uct_ep_invalidate_params_t params = {};
    struct mlx5_cqe64 *cqe;
    unsigned old_cq_ci;
    uint16_t old_logical_ci, cqe_pi;
    ssize_t initial_reads_available, initial_reads_completed;

    if (iface->super.super.config.max_inl_cqe[UCT_IB_DIR_TX] < length) {
        UCS_TEST_SKIP_R("TX inline response is unavailable");
    }

    flush();
    memset(localbuf.ptr(), 0, length);
    memset(remotebuf.ptr(), 0xa5, length);
    completion.uct.func     = completion_cb;
    completion.uct.count    = 1;
    completion.uct.status   = UCS_OK;
    initial_reads_available = iface->super.tx.reads_available;
    initial_reads_completed = iface->super.tx.reads_completed;

    uct_iface_progress_disable(m_e1->iface(), UCT_PROGRESS_SEND);
    ep_cleanup_guard cleanup(m_e1);
    old_cq_ci = cq->cq_ci;
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_get_bcopy(m_e1->ep(0), (uct_unpack_callback_t)memcpy,
                               localbuf.ptr(), length, remotebuf.addr(),
                               remotebuf.rkey(), &completion.uct));
    old_logical_ci = logical_ci(ep);
    ASSERT_EQ(1ul, ucs_queue_length(&txqp->outstanding));
    ASSERT_LT(iface->super.tx.reads_available, initial_reads_available);

    cqe = wait_for_cqe(cq, old_cq_ci);
    ASSERT_FALSE(cqe_is_hw_owned(cq, cqe, old_cq_ci));
    ucs_memory_cpu_load_fence();
    ASSERT_FALSE(cqe_is_error_or_zipped(cqe->op_own));
    ASSERT_TRUE(cqe->op_own &
                (MLX5_INLINE_SCATTER_32 | MLX5_INLINE_SCATTER_64));
    cqe_pi = ntohs(cqe->wqe_counter);

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;
    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), &params));

    EXPECT_GT(uct_iface_progress(m_e1->iface()), 0u);
    EXPECT_EQ(old_cq_ci + 1, cq->cq_ci);
    EXPECT_EQ(0, memcmp(localbuf.ptr(), remotebuf.ptr(), length));
    EXPECT_TRUE(ucs_queue_is_empty(&txqp->outstanding));
    EXPECT_EQ(1, completion.uct.count);
    EXPECT_EQ(0u, completion.callback_count);
    EXPECT_EQ(initial_reads_available, iface->super.tx.reads_available);
    EXPECT_EQ(initial_reads_completed, iface->super.tx.reads_completed);
    EXPECT_EQ(old_logical_ci, logical_ci(ep));
    EXPECT_EQ(cqe_pi, txwq->hw_ci);
}

UCS_TEST_SKIP_COND_P(test_rc_mlx5_late_cqe,
                     error_cqe_cleanup_before_callback,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY))
{
    uct_rc_mlx5_iface_common_t *iface =
            reinterpret_cast<uct_rc_mlx5_iface_common_t*>(m_e1->iface());
    uct_rc_mlx5_base_ep_t *ep    = reinterpret_cast<uct_rc_mlx5_base_ep_t*>(
            m_e1->ep(0));
    uct_base_iface_t *base_iface = &iface->super.super.super;
    uct_ib_mlx5_txwq_t *txwq     = &ep->tx.wq;
    uct_rc_txqp_t *txqp          = &ep->super.txqp;
    uct_ib_mlx5_cq_t *cq         = &iface->cq[UCT_IB_DIR_TX];
    const size_t length          = 32;
    mapped_buffer localbuf(length, 0ul, *m_e1);
    mapped_buffer remotebuf(length, 0ul, *m_e2);
    tracked_completion completion = {};
    error_context context         = {};
    struct mlx5_cqe64 error_cqe;
    struct mlx5_cqe64 *cqe;
    uct_error_handler_t saved_err_handler;
    void *saved_err_handler_arg;
    unsigned old_cq_ci;
    uint16_t cqe_pi;
    int16_t expected_available;

    flush();
    completion.uct.func   = completion_cb;
    completion.uct.count  = 1;
    completion.uct.status = UCS_OK;

    uct_iface_progress_disable(m_e1->iface(), UCT_PROGRESS_SEND);
    ep_cleanup_guard cleanup(m_e1);
    old_cq_ci = cq->cq_ci;
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_get_bcopy(m_e1->ep(0), (uct_unpack_callback_t)memcpy,
                               localbuf.ptr(), length, remotebuf.addr(),
                               remotebuf.rkey(), &completion.uct));
    ASSERT_EQ(1ul, ucs_queue_length(&txqp->outstanding));

    cqe = wait_for_cqe(cq, old_cq_ci);
    ASSERT_FALSE(cqe_is_hw_owned(cq, cqe, old_cq_ci));
    ucs_memory_cpu_load_fence();
    ASSERT_FALSE(cqe_is_error_or_zipped(cqe->op_own));
    cqe_pi = ntohs(cqe->wqe_counter);

    saved_err_handler     = base_iface->err_handler;
    saved_err_handler_arg = base_iface->err_handler_arg;
    context.ep                  = ep;
    context.completion          = &completion;
    base_iface->err_handler     = observe_error_cb;
    base_iface->err_handler_arg = &context;

    error_cqe         = make_error_cqe(txwq->super.qp_num, cqe_pi);
    error_cqe.op_own |= cqe->op_own & MLX5_CQE_OWNER_MASK;
    memcpy(cqe, &error_cqe, sizeof(*cqe));
    uct_ib_mlx5_check_completion_with_err(&iface->super.super, cq, cqe);

    base_iface->err_handler     = saved_err_handler;
    base_iface->err_handler_arg = saved_err_handler_arg;
    expected_available          = txwq->bb_max -
                                  (txwq->prev_sw_pi - cqe_pi);

    EXPECT_EQ(1u, context.callback_count);
    EXPECT_EQ(cqe_pi, context.observed_hw_ci);
    EXPECT_EQ(0u, context.observed_outstanding);
    EXPECT_EQ(expected_available, context.observed_available);
    EXPECT_EQ(1u, context.observed_completion_count);
    EXPECT_TRUE(ucs_queue_is_empty(&txqp->outstanding));
    EXPECT_EQ(0, completion.uct.count);
    EXPECT_EQ(UCS_ERR_CANCELED, completion.uct.status);
    EXPECT_EQ(1u, completion.callback_count);
}

UCS_TEST_SKIP_COND_P(test_rc_mlx5_late_cqe, error_cqe_state_before_callback,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY))
{
    uct_rc_mlx5_iface_common_t *iface =
            reinterpret_cast<uct_rc_mlx5_iface_common_t*>(m_e1->iface());
    uct_rc_mlx5_base_ep_t *ep    = reinterpret_cast<uct_rc_mlx5_base_ep_t*>(
            m_e1->ep(0));
    uct_base_iface_t *base_iface = &iface->super.super.super;
    uct_ib_mlx5_txwq_t *txwq     = &ep->tx.wq;
    uct_rc_txqp_t *txqp          = &ep->super.txqp;
    uct_ib_mlx5_cq_t *cq         = &iface->cq[UCT_IB_DIR_TX];
    const size_t length          = 32;
    mapped_buffer localbuf(length, 0ul, *m_e1);
    mapped_buffer remotebuf(length, 0ul, *m_e2);
    tracked_completion completion     = {};
    uct_ep_invalidate_params_t params = {};
    error_context context             = {};
    struct mlx5_cqe64 error_cqe;
    struct mlx5_cqe64 *cqe;
    uct_error_handler_t saved_err_handler;
    void *saved_err_handler_arg;
    unsigned old_cq_ci;
    uint16_t old_hw_ci, old_logical_ci, cqe_pi;
    int16_t post_available, observed_available;
    signed post_cq_available, observed_cq_available;
    ssize_t initial_reads_available;
    ssize_t post_reads_available, post_reads_completed;
    ssize_t observed_reads_available, observed_reads_completed;
    uint8_t saved_ep_flags;
    int16_t saved_fc_wnd;
#ifdef ENABLE_STATS
    uint64_t saved_fc_wnd_stat;
#endif
    bool outstanding_preserved;
#if UCS_ENABLE_ASSERT
    uint8_t saved_txwq_flags;
#endif

    flush();
    completion.uct.func     = completion_cb;
    completion.uct.count    = 1;
    completion.uct.status   = UCS_OK;
    initial_reads_available = iface->super.tx.reads_available;

    uct_iface_progress_disable(m_e1->iface(), UCT_PROGRESS_SEND);
    ep_cleanup_guard cleanup(m_e1);
    old_cq_ci = cq->cq_ci;
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_get_bcopy(m_e1->ep(0), (uct_unpack_callback_t)memcpy,
                               localbuf.ptr(), length, remotebuf.addr(),
                               remotebuf.rkey(), &completion.uct));
    ASSERT_EQ(1ul, ucs_queue_length(&txqp->outstanding));
    ASSERT_LT(iface->super.tx.reads_available, initial_reads_available);

    cqe = wait_for_cqe(cq, old_cq_ci);
    ASSERT_FALSE(cqe_is_hw_owned(cq, cqe, old_cq_ci));
    ucs_memory_cpu_load_fence();
    ASSERT_FALSE(cqe_is_error_or_zipped(cqe->op_own));

    cqe_pi               = ntohs(cqe->wqe_counter);
    old_hw_ci            = txwq->hw_ci;
    old_logical_ci       = logical_ci(ep);
    post_available       = txqp->available;
    post_cq_available    = iface->super.tx.cq_available;
    post_reads_available = iface->super.tx.reads_available;
    post_reads_completed = iface->super.tx.reads_completed;

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;
    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), &params));
    ASSERT_EQ(old_logical_ci, txwq->ft_ci);

    saved_err_handler     = base_iface->err_handler;
    saved_err_handler_arg = base_iface->err_handler_arg;
    saved_ep_flags        = ep->super.flags;
    saved_fc_wnd          = ep->super.fc.fc_wnd;
#ifdef ENABLE_STATS
    saved_fc_wnd_stat = UCS_STATS_GET_COUNTER(ep->super.fc.stats,
                                              UCT_RC_FC_STAT_FC_WND);
#endif
#if UCS_ENABLE_ASSERT
    saved_txwq_flags = txwq->flags;
#endif

    context.ep                  = ep;
    base_iface->err_handler     = observe_error_cb;
    base_iface->err_handler_arg = &context;

    error_cqe         = make_error_cqe(txwq->super.qp_num, cqe_pi);
    error_cqe.op_own |= cqe->op_own & MLX5_CQE_OWNER_MASK;
    memcpy(cqe, &error_cqe, sizeof(*cqe));
    uct_ib_mlx5_check_completion_with_err(&iface->super.super, cq, cqe);

    observed_cq_available    = iface->super.tx.cq_available;
    observed_available       = txqp->available;
    observed_reads_available = iface->super.tx.reads_available;
    observed_reads_completed = iface->super.tx.reads_completed;
    outstanding_preserved    = ucs_queue_length(&txqp->outstanding) == 1;

    base_iface->err_handler     = saved_err_handler;
    base_iface->err_handler_arg = saved_err_handler_arg;
    ep->super.flags             = saved_ep_flags;
    ep->super.fc.fc_wnd         = saved_fc_wnd;
#ifdef ENABLE_STATS
    UCS_STATS_SET_COUNTER(ep->super.fc.stats, UCT_RC_FC_STAT_FC_WND,
                          saved_fc_wnd_stat);
#endif
#if UCS_ENABLE_ASSERT
    txwq->flags = saved_txwq_flags;
#endif

    EXPECT_EQ(1u, context.callback_count);
    EXPECT_EQ(cqe_pi, context.observed_hw_ci);
    EXPECT_EQ(1u, context.observed_outstanding);
    EXPECT_EQ(old_cq_ci + 1, cq->cq_ci);
    EXPECT_EQ(cqe_pi, txwq->hw_ci);
    EXPECT_EQ(old_logical_ci, logical_ci(ep));
    EXPECT_EQ(old_logical_ci, txwq->ft_ci);
    EXPECT_EQ(post_cq_available + (uint16_t)(cqe_pi - old_hw_ci),
              observed_cq_available);
    EXPECT_EQ(post_available, observed_available);
    EXPECT_EQ(post_reads_available, observed_reads_available);
    EXPECT_EQ(post_reads_completed, observed_reads_completed);
    EXPECT_TRUE(outstanding_preserved);
    EXPECT_EQ(1, completion.uct.count);
    EXPECT_EQ(0u, completion.callback_count);
}

UCS_TEST_SKIP_COND_P(test_rc_mlx5_late_cqe, flush_after_no_completions,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    uct_rc_mlx5_iface_common_t *iface =
            reinterpret_cast<uct_rc_mlx5_iface_common_t*>(m_e1->iface());
    uct_rc_mlx5_base_ep_t *ep = reinterpret_cast<uct_rc_mlx5_base_ep_t*>(
            m_e1->ep(0));
    uct_ib_mlx5_txwq_t *txwq = &ep->tx.wq;
    uct_rc_txqp_t *txqp      = &ep->super.txqp;
    mapped_buffer sendbuf(32, 0ul, *m_e1);
    mapped_buffer recvbuf(32, 0ul, *m_e2);
    tracked_completion operation_completion = {};
    tracked_completion flush_completion     = {};
    uct_ep_invalidate_params_t params        = {};

    flush();
    operation_completion.uct.func   = completion_cb;
    operation_completion.uct.count  = 1;
    operation_completion.uct.status = UCS_OK;
    flush_completion.uct.func       = completion_cb;
    flush_completion.uct.count      = 1;
    flush_completion.uct.status     = UCS_OK;

    uct_iface_progress_disable(m_e1->iface(), UCT_PROGRESS_SEND);
    ep_cleanup_guard cleanup(m_e1);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(),
                            m_e1->iface_attr().cap.put.max_iov);
    ASSERT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(m_e1->ep(0), iov, iovcnt, recvbuf.addr(),
                               recvbuf.rkey(), &operation_completion.uct));

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;
    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), &params));

    const uint16_t sw_pi        = txwq->sw_pi;
    const uint16_t prev_sw_pi   = txwq->prev_sw_pi;
    const uint16_t sig_pi       = txwq->sig_pi;
    const uint16_t unsignaled   = txqp->unsignaled;
    const int16_t available     = txqp->available;
    const signed cq_available   = iface->super.tx.cq_available;
    const size_t outstanding    = ucs_queue_length(&txqp->outstanding);
    const uint8_t mlx5_ep_flags = ep->flags;
    const uint8_t rc_ep_flags   = ep->super.flags;

    ASSERT_GT(outstanding, 0ul);
    EXPECT_EQ(UCS_ERR_CANCELED,
              uct_ep_flush(m_e1->ep(0), 0, &flush_completion.uct));
    EXPECT_EQ(1, flush_completion.uct.count);
    EXPECT_EQ(UCS_OK, flush_completion.uct.status);
    EXPECT_EQ(0u, flush_completion.callback_count);
    EXPECT_EQ(1, operation_completion.uct.count);
    EXPECT_EQ(UCS_OK, operation_completion.uct.status);
    EXPECT_EQ(0u, operation_completion.callback_count);
    EXPECT_EQ(sw_pi, txwq->sw_pi);
    EXPECT_EQ(prev_sw_pi, txwq->prev_sw_pi);
    EXPECT_EQ(sig_pi, txwq->sig_pi);
    EXPECT_EQ(unsignaled, txqp->unsignaled);
    EXPECT_EQ(available, txqp->available);
    EXPECT_EQ(cq_available, iface->super.tx.cq_available);
    EXPECT_EQ(outstanding, ucs_queue_length(&txqp->outstanding));
    EXPECT_EQ(mlx5_ep_flags, ep->flags);
    EXPECT_EQ(rc_ep_flags, ep->super.flags);
}

UCS_TEST_SKIP_COND_P(test_rc_mlx5_late_cqe, ep_check_after_no_completions,
                     !check_caps(UCT_IFACE_FLAG_EP_CHECK))
{
    uct_rc_mlx5_iface_common_t *iface =
            reinterpret_cast<uct_rc_mlx5_iface_common_t*>(m_e1->iface());
    uct_rc_mlx5_base_ep_t *ep = reinterpret_cast<uct_rc_mlx5_base_ep_t*>(
            m_e1->ep(0));
    uct_ib_mlx5_txwq_t *txwq = &ep->tx.wq;
    uct_rc_txqp_t *txqp      = &ep->super.txqp;
    tracked_completion check_completion = {};
    uct_ep_invalidate_params_t params    = {};

    flush();
    check_completion.uct.func   = completion_cb;
    check_completion.uct.count  = 1;
    check_completion.uct.status = UCS_OK;

    uct_iface_progress_disable(m_e1->iface(), UCT_PROGRESS_SEND);
    ep_cleanup_guard cleanup(m_e1);
    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;
    ASSERT_UCS_OK(uct_ep_invalidate(m_e1->ep(0), &params));
    ASSERT_TRUE(ucs_arbiter_group_is_empty(&ep->super.arb_group));
    ASSERT_FALSE(ucs_arbiter_group_is_scheduled(&ep->super.arb_group));

    const signed saved_cq_available = iface->super.tx.cq_available;
    iface->super.tx.cq_available    = 0;

    const uint16_t sw_pi        = txwq->sw_pi;
    const uint16_t prev_sw_pi   = txwq->prev_sw_pi;
    const uint16_t sig_pi       = txwq->sig_pi;
    const uint16_t unsignaled   = txqp->unsignaled;
    const int16_t available     = txqp->available;
    const size_t outstanding    = ucs_queue_length(&txqp->outstanding);
    const uint8_t mlx5_ep_flags = ep->flags;
    const uint8_t rc_ep_flags   = ep->super.flags;

    EXPECT_EQ(UCS_ERR_CANCELED,
              uct_ep_check(m_e1->ep(0), 0, &check_completion.uct));
    EXPECT_EQ(1, check_completion.uct.count);
    EXPECT_EQ(UCS_OK, check_completion.uct.status);
    EXPECT_EQ(0u, check_completion.callback_count);
    EXPECT_EQ(sw_pi, txwq->sw_pi);
    EXPECT_EQ(prev_sw_pi, txwq->prev_sw_pi);
    EXPECT_EQ(sig_pi, txwq->sig_pi);
    EXPECT_EQ(unsignaled, txqp->unsignaled);
    EXPECT_EQ(available, txqp->available);
    EXPECT_EQ(0, iface->super.tx.cq_available);
    EXPECT_EQ(outstanding, ucs_queue_length(&txqp->outstanding));
    EXPECT_EQ(mlx5_ep_flags, ep->flags);
    EXPECT_EQ(rc_ep_flags, ep->super.flags);
    EXPECT_TRUE(ucs_arbiter_group_is_empty(&ep->super.arb_group));
    EXPECT_FALSE(ucs_arbiter_group_is_scheduled(&ep->super.arb_group));

    iface->super.tx.cq_available = saved_cq_available;
}

_UCT_INSTANTIATE_TEST_CASE(test_rc_mlx5_late_cqe, rc_mlx5)


class test_gga_mlx5_invalidate : public test_rc {
};

UCS_TEST_P(test_gga_mlx5_invalidate, no_completions_unsupported)
{
    mapped_buffer sendbuf(64, 0ul, *m_e1);
    mapped_buffer recvbuf(64, 0ul, *m_e2);
    uct_ep_invalidate_params_t params = {};
    uct_completion_t comp             = {};

    params.field_mask = UCT_EP_INVALIDATE_PARAM_FIELD_FLAGS;
    params.flags      = UCT_EP_INVALIDATE_FLAG_NO_COMPLETIONS;

    ASSERT_EQ(UCS_ERR_UNSUPPORTED, uct_ep_invalidate(m_e1->ep(0), &params));

    comp.func  = [](uct_completion_t*) {};
    comp.count = 1;
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(),
                            m_e1->iface_attr().cap.put.max_iov);
    ASSERT_UCS_OK_OR_INPROGRESS(
            uct_ep_put_zcopy(m_e1->ep(0), iov, iovcnt, recvbuf.addr(),
                             recvbuf.rkey(), &comp));
    flush();
}

_UCT_INSTANTIATE_TEST_CASE(test_gga_mlx5_invalidate, gga_mlx5)

#endif


class test_rc_max_wr : public test_rc {
protected:
    virtual void init() {
        ucs_status_t status1, status2;
        status1 = uct_config_modify(m_iface_config, "RC_VERBS_TX_MAX_WR", "32");
        status2 = uct_config_modify(m_iface_config, "RC_TX_MAX_BB", "32");
        if (status1 != UCS_OK && status2 != UCS_OK) {
            UCS_TEST_ABORT("Error: cannot set rc max wr/bb");
        }
        test_rc::init();
    }
};

/* Check that max_wr stops from sending */
UCS_TEST_SKIP_COND_P(test_rc_max_wr, send_limit,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    /* first 32 messages should be OK */
    send_am_messages(m_e1, 32, UCS_OK);

    /* next message - should fail */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_max_wr)


class test_rc_iface_flush_remote : public uct_test {
protected:
    entity *m_e1;
    entity *m_e2;
    entity *m_entity_flush_rkey;
    int m_err_count;

public:
    int rc_iface_flush_rkey_enabled(entity *e)
    {
        uct_rc_iface_t *rc_iface = ucs_derived_of(e->iface(), uct_rc_iface_t);
        return uct_rc_iface_flush_rkey_enabled(rc_iface);
    }

    int rc_iface_mr_id(entity *e)
    {
        uct_rc_iface_t *rc_iface = ucs_derived_of(e->iface(), uct_rc_iface_t);
        uct_ib_md_t *md          = uct_ib_iface_md(&rc_iface->super);
        return uct_ib_md_get_atomic_mr_id(md);
    }

    uct_iface_params_t iface_params()
    {
        uct_iface_params_t params = {};

        params.field_mask      = UCT_IFACE_PARAM_FIELD_ERR_HANDLER       |
                                 UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG   |
                                 UCT_IFACE_PARAM_FIELD_OPEN_MODE         |
                                 UCT_IFACE_PARAM_FIELD_FEATURES;
        params.open_mode       = UCT_IFACE_OPEN_MODE_DEVICE;
        params.err_handler_arg = &m_err_count,
        params.err_handler     =
            [](void *arg, uct_ep_h ep, ucs_status_t status) {
                (*reinterpret_cast<int*>(arg))++;
                return UCS_OK;
        };
        params.features        = UCT_IFACE_FEATURE_PUT;

        return params;
    }

    void init()
    {
        uct_test::init();

        m_err_count               = 0;
        uct_iface_params_t params = iface_params();
        m_e1                      = uct_test::create_entity(params);
        params                    = iface_params();
        m_e2                      = uct_test::create_entity(params);
        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        params.features    |= UCT_IFACE_FEATURE_FLUSH_REMOTE;
        m_entity_flush_rkey = uct_test::create_entity(params);

        m_entities.push_back(m_e1);
        m_entities.push_back(m_e2);
        m_entities.push_back(m_entity_flush_rkey);
    }

    using map_size_t = std::map<std::string, std::pair<size_t, size_t>>;

    void check_sizes(entity *e, const map_size_t &sizes)
    {
        auto it = sizes.find(GetParam()->tl_name);
        ASSERT_NE(sizes.end(), it);

        EXPECT_EQ(it->second.first, e->iface_attr().ep_addr_len);
        EXPECT_EQ(it->second.second, e->iface_attr().iface_addr_len);
    }
};

UCS_TEST_P(test_rc_iface_flush_remote, size_no_flush_remote)
{
    map_size_t sizes = {
        {"rc_mlx5", {7, 1}},
        {"dc_mlx5", {0, 5}},
        {"rc_verbs", {7, 0}},
        {"gga_mlx5", {7, 8}},
    };
    check_sizes(m_e1, sizes);
}

UCS_TEST_P(test_rc_iface_flush_remote, size_flush_remote)
{
    int flush_rkey_enabled = rc_iface_flush_rkey_enabled(m_entity_flush_rkey);
    int mr_id              = rc_iface_mr_id(m_entity_flush_rkey);
    map_size_t sizes = {
        {"rc_mlx5", {flush_rkey_enabled ? 10 : 7, 1}},
        {"dc_mlx5", {0, flush_rkey_enabled ? 7 : 5}},
        {"rc_verbs", {flush_rkey_enabled || (mr_id != 0) ? 7 : 4, 0}},
        {"gga_mlx5", {7, 8}},
    };
    check_sizes(m_entity_flush_rkey, sizes);
}

UCS_TEST_SKIP_COND_P(test_rc_iface_flush_remote, put_fence_no_flush_remote,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY),
                     "IB_PCI_RELAXED_ORDERING?=try")
{
    mapped_buffer sendbuf(64, 0ul, *m_e1);
    mapped_buffer recvbuf(64, 0ul, *m_e2);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), 64, sendbuf.memh(),
                            1);

    uct_completion_t comp;
    comp.func   = [](uct_completion_t*) {};
    comp.count  = 1;
    comp.status = UCS_OK;

    // Trigger the use of atomic key, PUT fails with invalid atomic_mr_offset
    ASSERT_UCS_OK(uct_ep_fence(m_e1->ep(0), 0));
    EXPECT_EQ(UCS_INPROGRESS,
              uct_ep_put_zcopy(m_e1->ep(0), iov, iovcnt, recvbuf.addr(),
                               recvbuf.rkey(), &comp));
    wait_for_value(&comp.count, 0, true);
    EXPECT_EQ(0, comp.count);
    EXPECT_EQ(0, m_err_count);
}

UCT_INSTANTIATE_RC_DC_GGA_TEST_CASE(test_rc_iface_flush_remote)


class test_rc_get_limit : public test_rc {
public:
    struct am_completion_t {
        uct_completion_t uct;
        uct_ep_h         ep;
        int              cb_count;
    };

    test_rc_get_limit() {
        m_num_get_bytes = 8 * UCS_KBYTE + 557; // some non power of 2 value
        modify_config("RC_TX_NUM_GET_BYTES",
                      ucs::to_string(m_num_get_bytes).c_str());
        m_max_get_zcopy = 4096;
        modify_config("RC_MAX_GET_ZCOPY",
                      ucs::to_string(m_max_get_zcopy).c_str());
        if (!RUNNING_ON_VALGRIND) {
            /* Valgrind already has special small value for this */
            modify_config("RC_TX_QUEUE_LEN", "32");
        }

        if (!ucs::skip_hw_tm_offload()) {
            modify_config("RC_TM_ENABLE", "y", SETENV_IF_NOT_EXIST);
        }

        m_comp.func   = NULL;
        m_comp.count  = 300000; // some big value to avoid func invocation
        m_comp.status = UCS_OK;

        stats_activate();
    }

    ~test_rc_get_limit()
    {
        stats_restore();
    }

#ifdef ENABLE_STATS
    uint64_t get_no_reads_stat_counter(entity *e) {
        uct_rc_iface_t *iface = ucs_derived_of(e->iface(), uct_rc_iface_t);

        return UCS_STATS_GET_COUNTER(iface->stats, UCT_RC_IFACE_STAT_NO_READS);
    }
#endif

    ssize_t reads_available(entity *e) {
        return rc_iface(e)->tx.reads_available;
    }

    void post_max_reads(entity *e, const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf) {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                                sendbuf.memh(), e->iface_attr().cap.get.max_iov);

        int i = 0;
        ucs_status_t status;
        do {
            if (i++ % 2) {
                status = uct_ep_get_zcopy(e->ep(0), iov, iovcnt, recvbuf.addr(),
                                          recvbuf.rkey(), &m_comp);
            } else {
                status = uct_ep_get_bcopy(e->ep(0), (uct_unpack_callback_t)memcpy,
                                          sendbuf.ptr(), sendbuf.length(),
                                          recvbuf.addr(), recvbuf.rkey(), &m_comp);
            }
        } while (status == UCS_INPROGRESS);

        EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
        EXPECT_GE(0u, reads_available(e));
    }

    void add_pending_ams(pending_send_request_t *reqs, int num_reqs) {
        for (int i = 0; i < num_reqs; ++i) {
            reqs[i].uct.func = pending_cb_send_am;
            reqs[i].ep       = m_e1->ep(0);
            reqs[i].cb_count = i;
            ASSERT_UCS_OK(uct_ep_pending_add(m_e1->ep(0), &reqs[i].uct, 0));
        }
    }

    static ucs_status_t pending_cb_send_am(uct_pending_req_t *self) {
        pending_send_request_t *req = ucs_container_of(self,
                                                       pending_send_request_t,
                                                       uct);

        return uct_ep_am_short(req->ep, AM_CHECK_ORDER_ID, req->cb_count,
                               NULL, 0);
    }

    static ucs_status_t am_handler_ordering(void *arg, void *data,
                                            size_t length, unsigned flags) {
        uint64_t *prev_sn = (uint64_t*)arg;
        uint64_t sn       = *(uint64_t*)data;

        EXPECT_LE(*prev_sn, sn);

        *prev_sn = sn;

        return UCS_OK;
    }

    static void get_comp_cb(uct_completion_t *self) {
        am_completion_t *comp = ucs_container_of(self, am_completion_t, uct);

        EXPECT_UCS_OK(self->status);

        ucs_status_t status = uct_ep_am_short(comp->ep, AM_CHECK_ORDER_ID,
                                              comp->cb_count, NULL, 0);
        EXPECT_TRUE(!UCS_STATUS_IS_ERR(status) ||
                    (status == UCS_ERR_NO_RESOURCE));
    }

    static size_t empty_pack_cb(void *dest, void *arg) {
        return 0ul;
    }

protected:
    static const uint8_t AM_CHECK_ORDER_ID = 1;
    unsigned             m_num_get_bytes;
    unsigned             m_max_get_zcopy;
    uct_completion_t     m_comp;
};

UCS_TEST_SKIP_COND_P(test_rc_get_limit, get_ops_limit,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_GET_BCOPY))
{
    mapped_buffer sendbuf(1024, 0ul, *m_e1);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);

    post_max_reads(m_e1, sendbuf, recvbuf);

#ifdef ENABLE_STATS
    EXPECT_GT(get_no_reads_stat_counter(m_e1), 0ul);
#endif

    // Check that it is possible to add to pending if get returns NO_RESOURCE
    // due to lack of get credits
    uct_pending_req_t pend_req;
    pend_req.func = NULL; // Make valgrind happy
    EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &pend_req, 0));
    uct_ep_pending_purge(m_e1->ep(0), NULL, NULL);

    flush();
    EXPECT_EQ(m_num_get_bytes, reads_available(m_e1));
}

// Check that get function fails for messages bigger than MAX_GET_ZCOPY value
UCS_TEST_SKIP_COND_P(test_rc_get_limit, get_size_limit,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    EXPECT_EQ(m_max_get_zcopy, m_e1->iface_attr().cap.get.max_zcopy);

    mapped_buffer buf(m_max_get_zcopy + 1, 0ul, *m_e1);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buf.ptr(), buf.length(), buf.memh(),
                            m_e1->iface_attr().cap.get.max_iov);

    scoped_log_handler wrap_err(wrap_errors_logger);
    ucs_status_t status = uct_ep_get_zcopy(m_e1->ep(0), iov, iovcnt,
                                           buf.addr(), buf.rkey(), &m_comp);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    flush();
    EXPECT_EQ(m_num_get_bytes, reads_available(m_e1));
}

// Check that get size value is trimmed by the actual maximum IB msg size
UCS_TEST_SKIP_COND_P(test_rc_get_limit, invalid_get_size,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    size_t max_ib_msg = uct_ib_iface_port_attr(&rc_iface(m_e1)->super)->max_msg_sz;

    modify_config("RC_MAX_GET_ZCOPY", ucs::to_string(max_ib_msg + 1).c_str());

    scoped_log_handler wrap_warn(hide_warns_logger);
    entity *e = uct_test::create_entity(0);
    m_entities.push_back(e);

    EXPECT_EQ(m_max_get_zcopy, m_e1->iface_attr().cap.get.max_zcopy);
}

// Check that gets resource counter is not affected/changed when the get
// function fails due to lack of some other resources.
UCS_TEST_SKIP_COND_P(test_rc_get_limit, post_get_no_res,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_AM_BCOPY))
{
    unsigned max_get_bytes = reads_available(m_e1);
    ucs_status_t status;

    do {
        status = send_am_message(m_e1, 0, 0);
    } while (status == UCS_OK);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
    EXPECT_EQ(max_get_bytes, reads_available(m_e1));

    mapped_buffer buf(1024, 0ul, *m_e1);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buf.ptr(), buf.length(), buf.memh(),
                            m_e1->iface_attr().cap.get.max_iov);

    status = uct_ep_get_zcopy(m_e1->ep(0), iov, iovcnt, buf.addr(), buf.rkey(),
                              &m_comp);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
    EXPECT_EQ(max_get_bytes, reads_available(m_e1));
#ifdef ENABLE_STATS
    EXPECT_EQ(get_no_reads_stat_counter(m_e1), 0ul);
#endif

    flush();
}

UCS_TEST_SKIP_COND_P(test_rc_get_limit, check_rma_ops,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_GET_BCOPY |
                                 UCT_IFACE_FLAG_PUT_SHORT |
                                 UCT_IFACE_FLAG_PUT_BCOPY |
                                 UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_AM_SHORT  |
                                 UCT_IFACE_FLAG_AM_BCOPY  |
                                 UCT_IFACE_FLAG_AM_ZCOPY))

{
    mapped_buffer sendbuf(1024, 0ul, *m_e1);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);

    post_max_reads(m_e1, sendbuf, recvbuf);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), 1, sendbuf.memh(), 1);
    uct_ep_h ep = m_e1->ep(0);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_put_short(ep, NULL, 0, 0, 0));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_put_bcopy(ep, NULL, NULL, 0, 0));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_put_zcopy(ep, iov, iovcnt, 0, 0,
                                                    NULL));

    if (check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64)) {
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP64));
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_CSWAP), FOP64));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic64_post(ep, UCT_ATOMIC_OP_ADD, 0, 0, 0));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic64_fetch(ep, UCT_ATOMIC_OP_ADD, 0, NULL, 0, 0,
                                        NULL));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic_cswap64(ep, 0, 0, 0, 0, NULL, NULL));
    }

    if (check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP32)) {
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP32));
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_CSWAP), FOP32));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic32_post(ep, UCT_ATOMIC_OP_ADD, 0, 0, 0));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic32_fetch(ep, UCT_ATOMIC_OP_ADD, 0, NULL, 0, 0,
                                        NULL));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic_cswap32(ep, 0, 0, 0, 0, NULL, NULL));
    }

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_am_short(ep, 0, 0, NULL, 0));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_am_bcopy(ep, 0, empty_pack_cb, NULL,
                                                   0));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_am_zcopy(ep, 0, NULL, 0, iov, iovcnt,
                                                   0, NULL));

    if (check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY)) {
        // we do not have partial tag offload support
        ASSERT_TRUE(check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT |
                               UCT_IFACE_FLAG_TAG_EAGER_ZCOPY |
                               UCT_IFACE_FLAG_TAG_RNDV_ZCOPY));

        EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_tag_eager_short(ep, 0ul, NULL, 0));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_tag_eager_bcopy(ep, 0ul, 0ul, empty_pack_cb, NULL, 0));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_tag_eager_zcopy(ep, 0ul, 0ul, iov, iovcnt, 0u, NULL));
        void *rndv_op = uct_ep_tag_rndv_zcopy(ep, 0ul, NULL, 0u, iov, iovcnt,
                                              0u, NULL);
        EXPECT_TRUE(UCS_PTR_IS_ERR(rndv_op));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_tag_rndv_request(ep, 0ul, NULL, 0u, 0u));
    }

    flush();
    EXPECT_EQ(m_num_get_bytes, reads_available(m_e1));
}

// Check that outstanding get ops purged gracefully when ep is closed.
// Also check that get resources taken by those ops are released.
UCS_TEST_SKIP_COND_P(test_rc_get_limit, get_zcopy_purge,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_GET_BCOPY))
{
    mapped_buffer sendbuf(1024, 0ul, *m_e1);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);

    post_max_reads(m_e1, sendbuf, recvbuf);

    scoped_log_handler hide_warn(hide_warns_logger);

    unsigned flags      = UCT_FLUSH_FLAG_CANCEL;
    ucs_time_t deadline = ucs::get_deadline();
    ucs_status_t status;
    do {
        ASSERT_EQ(1ul, m_e1->num_eps());
        status = uct_ep_flush(m_e1->ep(0), flags, NULL);
        progress();
        if ((flags & UCT_FLUSH_FLAG_CANCEL) && (status != UCS_ERR_NO_RESOURCE)) {
            ASSERT_UCS_OK_OR_INPROGRESS(status);
            flags = UCT_FLUSH_FLAG_LOCAL;
            continue;
        }
    } while (((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS)) &&
             (ucs_get_time() < deadline));

    m_e1->destroy_eps();
    flush();
    EXPECT_EQ(m_num_get_bytes, reads_available(m_e1));
}

// Check that it is not possible to send while not all pendings are dispatched
// yet. RDMA_READ resources are released in get function completion callbacks.
// Since in RC transports completions are handled after pending dispatch
// (to preserve ordering), RDMA_READ resources should be returned to iface
// in deferred manner.
UCS_TEST_SKIP_COND_P(test_rc_get_limit, ordering_pending,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_GET_BCOPY |
                                 UCT_IFACE_FLAG_AM_SHORT  |
                                 UCT_IFACE_FLAG_PENDING))
{
    volatile uint64_t sn = 0;
    ucs_status_t status;

    uct_iface_set_am_handler(m_e2->iface(), AM_CHECK_ORDER_ID,
                             am_handler_ordering, (void*)&sn, 0);

    mapped_buffer sendbuf(1024, 0ul, *m_e1);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);

    post_max_reads(m_e1, sendbuf, recvbuf);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE,
              uct_ep_am_short(m_e1->ep(0), AM_CHECK_ORDER_ID, 0, NULL, 0));

    const uint64_t num_pend = 3;
    pending_send_request_t reqs[num_pend];
    add_pending_ams(reqs, num_pend);

    do {
        progress();
        status = uct_ep_am_short(m_e1->ep(0), AM_CHECK_ORDER_ID, num_pend,
                                 NULL, 0);
    } while (status != UCS_OK);

    wait_for_value(&sn, num_pend, true);
    EXPECT_EQ(num_pend, sn);

    flush();
    EXPECT_EQ(m_num_get_bytes, reads_available(m_e1));
}

UCS_TEST_SKIP_COND_P(test_rc_get_limit, ordering_comp_cb,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_GET_BCOPY |
                                 UCT_IFACE_FLAG_AM_SHORT  |
                                 UCT_IFACE_FLAG_PENDING))
{
    volatile uint64_t sn    = 0;
    const uint64_t num_pend = 3;

    uct_iface_set_am_handler(m_e2->iface(), AM_CHECK_ORDER_ID,
                             am_handler_ordering, (void*)&sn, 0);

    mapped_buffer sendbuf(1024, 0ul, *m_e1);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);

    am_completion_t comp;
    comp.uct.func       = get_comp_cb;
    comp.uct.count      = 1;
    comp.uct.status     = UCS_OK;
    comp.ep             = m_e1->ep(0);
    comp.cb_count       = num_pend;
    ucs_status_t status = uct_ep_get_bcopy(m_e1->ep(0),
                                           (uct_unpack_callback_t)memcpy,
                                           sendbuf.ptr(), sendbuf.length(),
                                           recvbuf.addr(), recvbuf.rkey(),
                                           &comp.uct);
    ASSERT_FALSE(UCS_STATUS_IS_ERR(status));

    post_max_reads(m_e1, sendbuf, recvbuf);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE,
              uct_ep_am_short(m_e1->ep(0), AM_CHECK_ORDER_ID, 0, NULL, 0));

    pending_send_request_t reqs[num_pend];
    add_pending_ams(reqs, num_pend);

    wait_for_value(&sn, num_pend - 1, true);
    EXPECT_EQ(num_pend - 1, sn);

    flush();
    EXPECT_EQ(m_num_get_bytes, reads_available(m_e1));
}

UCT_INSTANTIATE_RC_DC_GGA_TEST_CASE(test_rc_get_limit)


class test_gga_get_zcopy_purge : public test_rc {
};

UCS_TEST_SKIP_COND_P(test_gga_get_zcopy_purge, get_zcopy_purge,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    mapped_buffer localbuf(64, 0ul, *m_e1);
    mapped_buffer remotebuf(64, 0ul, *m_e2);

    uct_completion_t comp;
    comp.func   = [](uct_completion_t*) {};
    comp.count  = 2;
    comp.status = UCS_OK;

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, localbuf.ptr(), localbuf.length(),
                            localbuf.memh(), m_e1->iface_attr().cap.get.max_iov);
    ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_get_zcopy(m_e1->ep(0), iov, iovcnt,
                                                 remotebuf.addr(),
                                                 remotebuf.rkey(), &comp));

    scoped_log_handler hide_warn(hide_warns_logger);
    m_e1->destroy_eps();

    EXPECT_EQ(1, comp.count);
    EXPECT_EQ(UCS_ERR_CANCELED, comp.status);
}

_UCT_INSTANTIATE_TEST_CASE(test_gga_get_zcopy_purge, gga_mlx5)


class test_rc_ece : public test_rc {
public:
    void init()
    {
        m_recv_count = 0;
        test_rc::init();
    }

    static size_t send_pack_cb(void *dest, void *arg)
    {
        size_t length = *(size_t*)arg;
        memset(dest, 0, length);
        return length;
    }

    static ucs_status_t
    recv_handler(void *arg, void *data, size_t length, unsigned flags)
    {
        EXPECT_EQ(*(size_t*)arg, length);
        ++m_recv_count;
        return UCS_OK;
    }

    void send_recv(uct_ep_h ep, entity *ent, size_t length, uint64_t ece)
    {
        EXPECT_EQ(ece, rc_iface(m_e1)->config.ece);

        uct_iface_set_am_handler(ent->iface(), 0, recv_handler, &length, 0);
        ssize_t packed_size = uct_ep_am_bcopy(ep, 0, send_pack_cb, &length, 0);
        ASSERT_EQ(length, packed_size);
        wait_for_value(&m_recv_count, (size_t)1, true);
    }

protected:
    static size_t m_recv_count;
};

size_t test_rc_ece::m_recv_count = 0;

UCS_TEST_SKIP_COND_P(test_rc_ece, ece_0, !check_caps(UCT_IFACE_FLAG_AM_BCOPY),
                     "RC_ECE=0")
{
    send_recv(m_e1->ep(0), m_e2, m_e1->iface_attr().cap.am.max_bcopy, 0);
}

UCS_TEST_SKIP_COND_P(test_rc_ece, ece_custom,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY), "RC_ECE=43223")
{
    send_recv(m_e1->ep(0), m_e2, m_e1->iface_attr().cap.am.max_bcopy, 43223);
}

UCS_TEST_SKIP_COND_P(test_rc_ece, ece_auto,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY), "RC_ECE=auto")
{
    send_recv(m_e1->ep(0), m_e2, m_e1->iface_attr().cap.am.max_bcopy,
              UCS_ULUNITS_AUTO);
}

UCS_TEST_SKIP_COND_P(test_rc_ece, ece_inf, !check_caps(UCT_IFACE_FLAG_AM_BCOPY),
                     "RC_ECE=inf")
{
    send_recv(m_e1->ep(0), m_e2, m_e1->iface_attr().cap.am.max_bcopy,
              UCS_ULUNITS_INF);
}

UCT_INSTANTIATE_RC_DC_TEST_CASE(test_rc_ece)

uint32_t test_rc_flow_control::m_am_rx_count = 0;

void test_rc_flow_control::init()
{
    /* For correct testing FC needs to be initialized during interface creation */
    if (UCS_OK != uct_config_modify(m_iface_config, "RC_FC_ENABLE", "y")) {
        UCS_TEST_ABORT("Error: cannot enable flow control");
    }
    test_rc::init();

    ucs_assert(rc_iface(m_e1)->config.fc_enabled);
    ucs_assert(rc_iface(m_e2)->config.fc_enabled);

    uct_iface_set_am_handler(m_e1->iface(), FLUSH_AM_ID, am_handler, NULL, 0);
    uct_iface_set_am_handler(m_e2->iface(), FLUSH_AM_ID, am_handler, NULL, 0);

}

void test_rc_flow_control::cleanup()
{
    /* Restore FC state to enabled, so iface cleanup will destroy the grant mpool */
    rc_iface(m_e1)->config.fc_enabled = 1;
    rc_iface(m_e2)->config.fc_enabled = 1;
    test_rc::cleanup();
}

void test_rc_flow_control::send_am_and_flush(entity *e, int num_msg)
{
    m_am_rx_count = 0;

    send_am_messages(e, num_msg - 1, UCS_OK);
    send_am_messages(e, 1, UCS_OK, FLUSH_AM_ID); /* send last msg with FLUSH id */
    wait_for_flag(&m_am_rx_count);
    EXPECT_EQ(m_am_rx_count, 1ul);
}

void test_rc_flow_control::validate_grant(entity *e)
{
    wait_for_flag(&get_fc_ptr(e)->fc_wnd);
    EXPECT_GT(get_fc_ptr(e)->fc_wnd, 0);
}

/* Check that FC window works as expected:
 * - If FC enabled, only 'wnd' messages can be sent in a row
 * - If FC is disabled 'wnd' does not limit senders flow  */
void test_rc_flow_control::test_general(int wnd, int soft_thresh,
                                        int hard_thresh, bool is_fc_enabled)
{
    set_fc_attributes(m_e1, is_fc_enabled, wnd, soft_thresh, hard_thresh);

    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, is_fc_enabled ?  UCS_ERR_NO_RESOURCE : UCS_OK);

    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);

    if (!is_fc_enabled) {
        /* Make valgrind happy, need to enable FC for proper cleanup */
        set_fc_attributes(m_e1, true, wnd, wnd, 1);
    }
    flush();
}

void test_rc_flow_control::wait_fc_hard_resend(entity *e)
{
}

void test_rc_flow_control::test_pending_grant(int16_t wnd)
{
    /* Block send capabilities of m_e2 for fc grant to be
     * added to the pending queue. */
    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    /* Now m_e1 should be blocked by FC window and FC grant
     * should be in pending queue of m_e2. */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);
    EXPECT_LE(get_fc_ptr(m_e1)->fc_wnd, 0);

    wait_fc_hard_resend(m_e1);

    /* Enable send capabilities of m_e2 and send short put message to force
     * pending queue dispatch. Can't send AM message for that, because it may
     * trigger reordering assert due to disable/enable entity hack. */
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 0);
    EXPECT_EQ(UCS_OK, uct_ep_put_short(m_e2->ep(0), NULL, 0, 0, 0));

    /* Check that m_e1 got grant */
    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);
}

void test_rc_flow_control::test_flush_fc_disabled()
{
    set_fc_disabled(m_e1);
    ucs_status_t status;

    /* If FC is disabled, wnd=0 should not prevent the flush */
    get_fc_ptr(m_e1)->fc_wnd = 0;
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_EQ(UCS_OK, status);

    /* send active message should be OK */
    get_fc_ptr(m_e1)->fc_wnd = 1;
    send_am_messages(m_e1, 1, UCS_OK);
    EXPECT_EQ(0, get_fc_ptr(m_e1)->fc_wnd);

    /* flush must have resources */
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_FALSE(UCS_STATUS_IS_ERR(status)) << ucs_status_string(status);
}

void test_rc_flow_control::test_pending_purge(int wnd, int num_pend_sends)
{
    pending_send_request_t reqs[num_pend_sends];

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    /* Now m2 ep should have FC grant message in the pending queue.
     * Add some user pending requests as well */
    for (int i = 0; i < num_pend_sends; i++) {
        reqs[i].uct.func    = NULL; /* make valgrind happy */
        reqs[i].purge_count = 0;
        EXPECT_EQ(uct_ep_pending_add(m_e2->ep(0), &reqs[i].uct, 0), UCS_OK);
    }
    uct_ep_pending_purge(m_e2->ep(0), purge_cb, NULL);

    for (int i = 0; i < num_pend_sends; i++) {
        EXPECT_EQ(1, reqs[i].purge_count);
    }
}


/* Check that FC window works as expected */
UCS_TEST_SKIP_COND_P(test_rc_flow_control, general_enabled,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    test_general(8, 4, 2, true);
}

UCS_TEST_SKIP_COND_P(test_rc_flow_control, general_disabled,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    test_general(8, 4, 2, false);
}

/* Test the scenario when ep is being destroyed while there is
 * FC grant message in the pending queue */
UCS_TEST_SKIP_COND_P(test_rc_flow_control, pending_only_fc,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    int wnd = 2;

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    m_e2->destroy_ep(0);
    ASSERT_TRUE(ucs_arbiter_is_empty(&rc_iface(m_e2)->tx.arbiter));
}

/* Check that user callback passed to uct_ep_pending_purge is not
 * invoked for FC grant message */
UCS_TEST_SKIP_COND_P(test_rc_flow_control, pending_purge,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    test_pending_purge(2, 5);
}

UCS_TEST_SKIP_COND_P(test_rc_flow_control, pending_grant,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    test_pending_grant(5);
}

UCS_TEST_SKIP_COND_P(test_rc_flow_control, fc_disabled_flush,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    test_flush_fc_disabled();
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_flow_control)


#ifdef ENABLE_STATS

void test_rc_flow_control_stats::test_general(int wnd, int soft_thresh,
                                              int hard_thresh)
{
    uint64_t v;

    set_fc_attributes(m_e1, true, wnd, soft_thresh, hard_thresh);

    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_NO_CRED);
    EXPECT_EQ(1ul, v);

    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_HARD_REQ);
    EXPECT_EQ(1ul, v);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
    flush();
}


UCS_TEST_SKIP_COND_P(test_rc_flow_control_stats, general,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    test_general(5, 2, 1);
}

UCS_TEST_SKIP_COND_P(test_rc_flow_control_stats, soft_request,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY))
{
    uint64_t v;
    int wnd = 8;
    int s_thresh = 4;
    int h_thresh = 1;

    set_fc_attributes(m_e1, true, wnd, s_thresh, h_thresh);
    send_am_and_flush(m_e1, wnd - (s_thresh - 1));

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_SOFT_REQ);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_RX_SOFT_REQ);
    EXPECT_EQ(1ul, v);

    send_am_and_flush(m_e2, wnd - (s_thresh - 1));
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_GRANT);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_TX_GRANT);
    EXPECT_EQ(1ul, v);
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_flow_control_stats)

#endif

#ifdef HAVE_MLX5_DV
class test_gga_fence_flags : public uct_test {
protected:
    void init()
    {
        uct_test::init();

        m_entity = uct_test::create_entity(0);
        m_entities.push_back(m_entity);
    }

    void cleanup()
    {
        m_iface.reset();
        uct_test::cleanup();
    }

    ucs_status_t open_iface(uct_ib_mlx5_dp_ordering_t dp_ordering,
                            bool relaxed_order, bool pci_atomics)
    {
        uct_ib_mlx5_md_t *md = ucs_derived_of(m_entity->md(), uct_ib_mlx5_md_t);
        uct_ib_device_t *dev      = &md->super.dev;
        uint8_t saved_dp_ordering = md->dp_ordering_cap_devx.rc;
        uint8_t saved_pci_fadd    = dev->pci_fadd_arg_sizes;
        uint8_t saved_pci_cswap   = dev->pci_cswap_arg_sizes;
        uint64_t saved_relaxed_order_mem_types =
                md->super.relaxed_order_mem_types;
        uct_iface_h tl_iface      = NULL;
        ucs_status_t status;

        m_iface.reset();
        md->dp_ordering_cap_devx.rc = dp_ordering;
        md->super.relaxed_order_mem_types =
                relaxed_order ? UCS_BIT(UCS_MEMORY_TYPE_HOST) : 0;
        dev->pci_fadd_arg_sizes     = pci_atomics ? sizeof(uint64_t) : 0;
        dev->pci_cswap_arg_sizes    = pci_atomics ? sizeof(uint64_t) : 0;

        status = uct_iface_open(m_entity->md(), m_entity->worker(),
                                &m_entity->iface_params(), m_iface_config,
                                &tl_iface);

        dev->pci_cswap_arg_sizes    = saved_pci_cswap;
        dev->pci_fadd_arg_sizes     = saved_pci_fadd;
        md->super.relaxed_order_mem_types =
                saved_relaxed_order_mem_types;
        md->dp_ordering_cap_devx.rc = saved_dp_ordering;

        if (status == UCS_OK) {
            m_iface.reset(tl_iface, uct_iface_close);
        }

        return status;
    }

    uct_rc_mlx5_iface_common_t *rc_iface()
    {
        return ucs_derived_of(m_iface.get(), uct_rc_mlx5_iface_common_t);
    }

    void
    check_enabled(uct_ib_mlx5_dp_ordering_t dp_ordering, uint8_t get_fence_flag)
    {
        EXPECT_EQ(dp_ordering, rc_iface()->config.dp_ordering_devx);
        EXPECT_NE(UCT_RC_FENCE_MODE_NONE, rc_iface()->super.config.fence_mode);
        EXPECT_EQ(UCT_IB_MLX5_WQE_CTRL_FLAG_STRONG_ORDER,
                  rc_iface()->config.put_fence_flag);
        EXPECT_EQ(get_fence_flag, rc_iface()->config.atomic_fence_flag);
    }

    void check_disabled(uct_ib_mlx5_dp_ordering_t dp_ordering)
    {
        EXPECT_EQ(dp_ordering, rc_iface()->config.dp_ordering_devx);
        EXPECT_EQ(UCT_RC_FENCE_MODE_NONE, rc_iface()->super.config.fence_mode);
        EXPECT_EQ(0, rc_iface()->config.put_fence_flag);
        EXPECT_EQ(0, rc_iface()->config.atomic_fence_flag);
    }

private:
    entity *m_entity = NULL;
    ucs::handle<uct_iface_h> m_iface;
};

UCS_TEST_P(test_gga_fence_flags, auto_ibta, "GGA_MLX5_AR_ENABLE=auto",
           "GGA_MLX5_FENCE=auto")
{
    ASSERT_UCS_OK(open_iface(UCT_IB_MLX5_DP_ORDERING_IBTA, false, false));
    check_enabled(UCT_IB_MLX5_DP_ORDERING_IBTA,
                  UCT_IB_MLX5_WQE_CTRL_FLAG_FENCE);
}

UCS_TEST_P(test_gga_fence_flags, weak_ibta, "GGA_MLX5_AR_ENABLE=auto",
           "GGA_MLX5_FENCE=weak")
{
    ASSERT_UCS_OK(open_iface(UCT_IB_MLX5_DP_ORDERING_IBTA, false, false));
    check_enabled(UCT_IB_MLX5_DP_ORDERING_IBTA,
                  UCT_IB_MLX5_WQE_CTRL_FLAG_FENCE);
}

UCS_TEST_P(test_gga_fence_flags, none_ibta, "GGA_MLX5_AR_ENABLE=auto",
           "GGA_MLX5_FENCE=none")
{
    ASSERT_UCS_OK(open_iface(UCT_IB_MLX5_DP_ORDERING_IBTA, false, false));
    check_disabled(UCT_IB_MLX5_DP_ORDERING_IBTA);
}

UCS_TEST_P(test_gga_fence_flags, auto_ooo, "GGA_MLX5_AR_ENABLE=auto",
           "GGA_MLX5_FENCE=auto")
{
    ASSERT_UCS_OK(open_iface(UCT_IB_MLX5_DP_ORDERING_OOO_RW, false, false));
    check_enabled(UCT_IB_MLX5_DP_ORDERING_OOO_RW,
                  UCT_IB_MLX5_WQE_CTRL_FLAG_STRONG_ORDER);
}

UCS_TEST_P(test_gga_fence_flags, none_ooo, "GGA_MLX5_AR_ENABLE=auto",
           "GGA_MLX5_FENCE=none")
{
    ASSERT_UCS_OK(open_iface(UCT_IB_MLX5_DP_ORDERING_OOO_RW, false, false));
    check_disabled(UCT_IB_MLX5_DP_ORDERING_OOO_RW);
}

UCS_TEST_P(test_gga_fence_flags, auto_relaxed_order, "GGA_MLX5_AR_ENABLE=auto",
           "GGA_MLX5_FENCE=auto")
{
    ASSERT_UCS_OK(open_iface(UCT_IB_MLX5_DP_ORDERING_IBTA, true, false));
    check_enabled(UCT_IB_MLX5_DP_ORDERING_IBTA,
                  UCT_IB_MLX5_WQE_CTRL_FLAG_FENCE);
}

UCS_TEST_P(test_gga_fence_flags, auto_pci_atomics, "GGA_MLX5_AR_ENABLE=auto",
           "GGA_MLX5_FENCE=auto")
{
    ASSERT_UCS_OK(open_iface(UCT_IB_MLX5_DP_ORDERING_IBTA, false, true));
    check_enabled(UCT_IB_MLX5_DP_ORDERING_IBTA,
                  UCT_IB_MLX5_WQE_CTRL_FLAG_FENCE);
}

_UCT_INSTANTIATE_TEST_CASE(test_gga_fence_flags, gga_mlx5)
#endif

test_uct_iface_attrs::attr_map_t test_rc_iface_attrs::get_num_iov() {
    if (has_transport("rc_mlx5") || has_transport("gga_mlx5")) {
        return get_num_iov_mlx5_common(0ul);
    } else {
        EXPECT_TRUE(has_transport("rc_verbs"));
        m_e->connect(0, *m_e, 0);
        uct_rc_verbs_ep_t *ep = ucs_derived_of(m_e->ep(0), uct_rc_verbs_ep_t);
        uint32_t max_sge = 0; // for gcc 10 -Og
        ASSERT_UCS_OK(uct_ib_qp_max_send_sge(ep->qp, &max_sge));

        attr_map_t iov_map;
        iov_map["put"] = iov_map["get"] = max_sge;
        iov_map["am"]  = max_sge - 1; // 1 iov reserved for am header
        return iov_map;
    }
}

test_uct_iface_attrs::attr_map_t
test_rc_iface_attrs::get_num_iov_mlx5_common(size_t av_size)
{
    attr_map_t iov_map;

#ifdef HAVE_MLX5_DV
    size_t rma_iov = has_transport("gga_mlx5") ? 1 :
                     // For RMA iovs can use all WQE space, remaining from
                     // control and remote address segments (and AV if relevant)
                     (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                      (sizeof(struct mlx5_wqe_raddr_seg) +
                       sizeof(struct mlx5_wqe_ctrl_seg) + av_size)) /
                     sizeof(struct mlx5_wqe_data_seg);

    iov_map["put"] = iov_map["get"] = rma_iov;

    // For am zcopy just small constant number of iovs is allowed
    // (to preserve some inline space for AM zcopy header)
    iov_map["am"]  = UCT_IB_MLX5_AM_ZCOPY_MAX_IOV;

#if IBV_HW_TM
    if (UCT_RC_MLX5_TM_ENABLED(ucs_derived_of(m_e->iface(),
                                              uct_rc_mlx5_iface_common_t))) {
        // For TAG eager zcopy iovs can use all WQE space, remaining from control
        // segment, TMH header (+ inline data segment) and AV (if relevant)
        iov_map["tag"] = (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                          (sizeof(struct mlx5_wqe_ctrl_seg) +
                           sizeof(struct mlx5_wqe_inl_data_seg) +
                           sizeof(struct ibv_tmh) + av_size)) /
                         sizeof(struct mlx5_wqe_data_seg);
    }
#endif // IBV_HW_TM
#endif // HAVE_MLX5_DV

    return iov_map;
}

UCS_TEST_P(test_rc_iface_attrs, iface_attrs)
{
    basic_iov_test();
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_iface_attrs)

class test_rc_keepalive : public test_uct_peer_failure {
public:
    uct_rc_iface_t* rc_iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_rc_iface_t);
    }

    virtual void disable_entity(entity *e) {
        rc_iface(e)->tx.cq_available = 0;
    }

    virtual void enable_entity(entity *e, unsigned cq_num = 128) {
        rc_iface(e)->tx.cq_available = cq_num;
    }
};

/* this test is quite tricky: it emulates missing iface resources
 * to force keepalive operation push into arbiter. after this
 * iface resources are restored, peer is killed and initiated processing
 * of arbiter operations.
 * we can't just call progress to initiate arbiter because there is
 * no completions, and we can't initiate completion by any operation
 * because it will produce failure (even in case if keepalive is not
 * called and test will pass even in case if keepalive doesn't work).
 */
UCS_TEST_SKIP_COND_P(test_rc_keepalive, pending,
                     !check_caps(UCT_IFACE_FLAG_EP_CHECK))
{
    ucs_status_t status;

    scoped_log_handler slh(wrap_errors_logger);
    flush();
    /* ensure that everything works as expected */
    EXPECT_EQ(0, m_err_count);

    /* regular ep_check operation should be completed successfully */
    status = uct_ep_check(ep0(), 0, NULL);
    ASSERT_UCS_OK(status);
    flush();
    EXPECT_EQ(0, m_err_count);

    /* emulate for lack of iface resources. after this all
     * send/keepalive/etc operations will not be processed */
    disable_entity(m_sender);

    /* try to send keepalive message: there are TX resources, but not CQ
     * resources. keepalive operation should be posted to pending queue */
    status = uct_ep_check(ep0(), 0, NULL);
    ASSERT_UCS_OK(status);

    inject_error();

    enable_entity(m_sender);

    /* initiate processing of pending operations: scheduled keepalive
     * operation should be processed & failed because peer is killed */
    ucs_arbiter_dispatch(&rc_iface(m_sender)->tx.arbiter, 1,
                         uct_rc_ep_process_pending, NULL);

    wait_for_flag(&m_err_count);
    EXPECT_EQ(1, m_err_count);
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_keepalive)


#ifdef HAVE_MLX5_DV

class test_rc_srq : public test_rc {
public:
    test_rc_srq() : m_buf8b(NULL), m_buf8k(NULL)
    {
    }

    void init()
    {
        test_rc::init();

        m_buf8b = new mapped_buffer(8, 0x1, *m_e1);
        m_buf8k = new mapped_buffer(8 * UCS_KBYTE, 0x2, *m_e1);
    }

    void connect()
    {
        test_rc::connect();

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
        m_e1->connect(1, *m_e2, 1);
        m_e2->connect(1, *m_e1, 1);
    }

    bool send(int ep, void *buf)
    {
        ssize_t status;

        status = uct_ep_am_bcopy(m_e1->ep(ep), 0, mapped_buffer::pack, buf, 0);
        if (status == UCS_ERR_NO_RESOURCE) {
            short_progress_loop();
            return false;
        } else if (status < 0) {
            ASSERT_UCS_OK((ucs_status_t)status);
        }

        return true;
    }

    void test_reorder() {
        unsigned i = 0;
        ucs_time_t deadline = ucs::get_deadline();
        while ((i < 10000) && (ucs_get_time() < deadline)) {
            if (send(0, m_buf8k) && send(1, m_buf8b)) {
                i++;
            }
        }
    }

    void cleanup() {
        delete m_buf8b;
        delete m_buf8k;
        test_rc::cleanup();
    }

protected:
    mapped_buffer *m_buf8b, *m_buf8k;
};

UCS_TEST_SKIP_COND_P(test_rc_srq, reorder_list,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY),
                     "RC_SRQ_TOPO?=list")
{
    test_reorder();
}

UCS_TEST_SKIP_COND_P(test_rc_srq, reorder_cyclic,
                     !check_caps(UCT_IFACE_FLAG_AM_BCOPY),
                     /* Disable DDP to allow cyclic SRQ */
                     "RC_MLX5_DDP_ENABLE?=n",
                     "DC_MLX5_DDP_ENABLE?=n",
                     "RC_SRQ_TOPO?=cyclic,cyclic_emulated")
{
    test_reorder();
}

UCT_INSTANTIATE_RC_DC_TEST_CASE(test_rc_srq);

#endif
