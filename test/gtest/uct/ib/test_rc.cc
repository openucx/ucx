/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

#include "test_rc.h"


#define UCT_RC_INSTANTIATE_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_mlx5)


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
                            sendbuf.memh(),
                            m_e1->iface_attr().cap.am.max_iov);
    // For _x transports several CQEs can be consumed per WQE, post less put zcopy
    // ops, so that flush would be sucessfull (otherwise flush will return
    // NO_RESOURCES and completion will not be added for it).
    for (int i = 0; i < cq_len / 3; i++) {
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

UCT_RC_INSTANTIATE_TEST_CASE(test_rc)


class test_rc_max_wr : public test_rc {
protected:
    virtual void init() {
        ucs_status_t status1, status2;
        status1 = uct_config_modify(m_iface_config, "TX_MAX_WR", "32");
        status2 = uct_config_modify(m_iface_config, "RC_TX_MAX_BB", "32");
        if (status1 != UCS_OK && status2 != UCS_OK) {
            UCS_TEST_ABORT("Error: cannot set rc max wr/bb");
        }
        test_rc::init();
    }
};

/* Check that max_wr stops from sending */
UCS_TEST_P(test_rc_max_wr, send_limit)
{
    /* first 32 messages should be OK */
    send_am_messages(m_e1, 32, UCS_OK);

    /* next message - should fail */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_max_wr)

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

void test_rc_flow_control::test_pending_grant(int wnd)
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

    /* Enable send capabilities of m_e2 and send AM message
     * to force pending queue dispatch */
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 0);
    send_am_messages(m_e2, 1, UCS_OK);

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
UCS_TEST_P(test_rc_flow_control, general_enabled)
{
    test_general(8, 4, 2, true);
}

UCS_TEST_P(test_rc_flow_control, general_disabled)
{
    test_general(8, 4, 2, false);
}

/* Test the scenario when ep is being destroyed while there is
 * FC grant message in the pending queue */
UCS_TEST_P(test_rc_flow_control, pending_only_fc)
{
    int wnd = 2;

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    m_e2->destroy_ep(0);
    ASSERT_TRUE(rc_iface(m_e2)->tx.arbiter.current == NULL);
}

/* Check that user callback passed to uct_ep_pending_purge is not
 * invoked for FC grant message */
UCS_TEST_P(test_rc_flow_control, pending_purge)
{
    test_pending_purge(2, 5);
}

UCS_TEST_P(test_rc_flow_control, pending_grant)
{
    test_pending_grant(5);
}

UCS_TEST_P(test_rc_flow_control, fc_disabled_flush)
{
    test_flush_fc_disabled();
}

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control)


#ifdef IBV_HW_TM
// TODO: Remove when declared in UCT
#define UCT_RC_MLX5_TAG_BCOPY_MAX     131072

size_t test_rc_mp_xrq::m_rx_counter = 0;

test_rc_mp_xrq::test_rc_mp_xrq() : m_hold_uct_desc(false),
                                   m_first_received(false),
                                   m_last_received(false)
{
    m_max_hdr        = sizeof(ibv_tmh) + sizeof(ibv_rvh);
    m_uct_comp.count = 512; // We do not need completion func to be invoked
    m_uct_comp.func  = NULL;
}

uct_rc_mlx5_iface_common_t* test_rc_mp_xrq::rc_mlx5_iface(entity &e)
{
    return ucs_derived_of(e.iface(), uct_rc_mlx5_iface_common_t);
}

void test_rc_mp_xrq::init()
{
    ucs_status_t status1 = uct_config_modify(m_iface_config,
                                             "RC_TM_MP_NUM_STRIDES", "8");
    ucs_status_t status2 = uct_config_modify(m_iface_config,
                                             "RC_TM_ENABLE", "y");
    if ((status1 != UCS_OK) || (status2 != UCS_OK)) {
        UCS_TEST_SKIP_R("No MP XRQ support");
    }

    uct_test::init();

    uct_iface_params params;
    params.field_mask  = UCT_IFACE_PARAM_FIELD_RX_HEADROOM     |
                         UCT_IFACE_PARAM_FIELD_OPEN_MODE       |
                         UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_CB  |
                         UCT_IFACE_PARAM_FIELD_HW_TM_EAGER_ARG |
                         UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_CB   |
                         UCT_IFACE_PARAM_FIELD_HW_TM_RNDV_ARG;

    // tl and dev names are taken from resources via GetParam, no need
    // to fill it here
    params.rx_headroom = 0;
    params.open_mode   = UCT_IFACE_OPEN_MODE_DEVICE;
    params.eager_cb    = unexp_eager;
    params.eager_arg   = reinterpret_cast<void*>(this);
    params.rndv_cb     = unexp_rndv;
    params.rndv_arg    = reinterpret_cast<void*>(this);

    entity *sender = uct_test::create_entity(params);
    m_entities.push_back(sender);

    entity *receiver = uct_test::create_entity(params);
    m_entities.push_back(receiver);

    sender->connect(0, *receiver, 0);

    uct_iface_set_am_handler(receiver->iface(), AM_ID, am_handler, this, 0);
}

void test_rc_mp_xrq::send_eager_bcopy(mapped_buffer *buf)
{
    ssize_t len = uct_ep_tag_eager_bcopy(sender().ep(0), 0x11,
                                         reinterpret_cast<uint64_t>(this),
                                         mapped_buffer::pack,
                                         reinterpret_cast<void*>(buf), 0);

    EXPECT_EQ(buf->length(), static_cast<size_t>(len));
}

void test_rc_mp_xrq::send_eager_zcopy(mapped_buffer *buf)
{
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buf->ptr(), buf->length(), buf->memh(),
                            sender().iface_attr().cap.tag.eager.max_iov);

    ucs_status_t status = uct_ep_tag_eager_zcopy(sender().ep(0), 0x11,
                                                 reinterpret_cast<uint64_t>(this),
                                                 iov, iovcnt, 0, &m_uct_comp);
    ASSERT_UCS_OK_OR_INPROGRESS(status);
}

void test_rc_mp_xrq::send_rndv_zcopy(mapped_buffer *buf)
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

void test_rc_mp_xrq::send_rndv_request(mapped_buffer *buf)
{
    size_t size = sender().iface_attr().cap.tag.rndv.max_hdr;
    void *hdr   = alloca(size);

    ASSERT_UCS_OK(uct_ep_tag_rndv_request(sender().ep(0), 0x11, hdr, size, 0));
}

void test_rc_mp_xrq::send_am_bcopy(mapped_buffer *buf)
{
    ssize_t len = uct_ep_am_bcopy(sender().ep(0), AM_ID, mapped_buffer::pack,
                                  reinterpret_cast<void*>(buf), 0);

    EXPECT_EQ(buf->length(), static_cast<size_t>(len));
}

void test_rc_mp_xrq::test_common(send_func sfunc, size_t num_segs,
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

ucs_status_t test_rc_mp_xrq::handle_uct_desc(void *data, unsigned flags)
{
    if ((flags & UCT_CB_PARAM_FLAG_DESC) && m_hold_uct_desc) {
        m_uct_descs.push_back(data);
        return UCS_INPROGRESS;
    }

    return UCS_OK;
}

ucs_status_t test_rc_mp_xrq::am_handler(void *arg, void *data, size_t length,
                                        unsigned flags)
{
   // These flags are intended for tag offload only
   EXPECT_FALSE(flags & (UCT_CB_PARAM_FLAG_MORE | UCT_CB_PARAM_FLAG_FIRST));

   m_rx_counter++;

   test_rc_mp_xrq *self = reinterpret_cast<test_rc_mp_xrq*>(arg);
   return self->handle_uct_desc(data, flags);
}

ucs_status_t test_rc_mp_xrq::unexp_handler(void *data, unsigned flags,
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

ucs_status_t test_rc_mp_xrq::unexp_eager(void *arg, void *data, size_t length,
                                         unsigned flags, uct_tag_t stag,
                                         uint64_t imm, void **context)
{
    test_rc_mp_xrq *self = reinterpret_cast<test_rc_mp_xrq*>(arg);

    m_rx_counter++;

    return self->unexp_handler(data, flags, imm, context);
}

ucs_status_t test_rc_mp_xrq::unexp_rndv(void *arg, unsigned flags,
                                        uint64_t stag, const void *header,
                                        unsigned header_length,
                                        uint64_t remote_addr, size_t length,
                                        const void *rkey_buf)
{
    EXPECT_TRUE(flags & UCT_CB_PARAM_FLAG_FIRST);
    EXPECT_FALSE(flags & UCT_CB_PARAM_FLAG_MORE);

    m_rx_counter++;

    return UCS_OK;
}

UCS_TEST_P(test_rc_mp_xrq, config)
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
    EXPECT_EQ(UCT_RC_MLX5_TAG_BCOPY_MAX - sizeof(ibv_tmh),
              attrs->cap.tag.eager.max_bcopy);

    // Max tag zcopy is limited by maximal IB message size
    EXPECT_EQ(uct_ib_iface_port_attr(&iface->super.super)->max_msg_sz - sizeof(ibv_tmh),
              attrs->cap.tag.eager.max_zcopy);

    // Maximal AM size should not exceed segment size, so it would always
    // arrive in one-fragment packet (with header it should be strictly less)
    EXPECT_LT(attrs->cap.am.max_bcopy, iface->super.super.config.seg_size);
    EXPECT_LT(attrs->cap.am.max_zcopy, iface->super.super.config.seg_size);
}

UCS_TEST_P(test_rc_mp_xrq, desc_release)
{
    m_hold_uct_desc = true; // We want to "hold" UCT memory descriptors
    std::pair<send_func, bool> sfuncs[5] = {
              std::make_pair(&test_rc_mp_xrq::send_eager_bcopy,  true),
              std::make_pair(&test_rc_mp_xrq::send_eager_zcopy,  true),
              std::make_pair(&test_rc_mp_xrq::send_rndv_zcopy,   false),
              std::make_pair(&test_rc_mp_xrq::send_rndv_request, false),
              std::make_pair(&test_rc_mp_xrq::send_am_bcopy,     false)
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

UCS_TEST_P(test_rc_mp_xrq, am)
{
    test_common(&test_rc_mp_xrq::send_am_bcopy, 1, 1, false);
}

UCS_TEST_P(test_rc_mp_xrq, bcopy_eager_only)
{
    test_common(&test_rc_mp_xrq::send_eager_bcopy, 1);
}

UCS_TEST_P(test_rc_mp_xrq, zcopy_eager_only)
{
    test_common(&test_rc_mp_xrq::send_eager_zcopy, 1);
}

UCS_TEST_P(test_rc_mp_xrq, bcopy_eager)
{
    test_common(&test_rc_mp_xrq::send_eager_bcopy, 5, 5);
}

UCS_TEST_P(test_rc_mp_xrq, zcopy_eager)
{
    test_common(&test_rc_mp_xrq::send_eager_zcopy, 5, 5);
}

UCS_TEST_P(test_rc_mp_xrq, rndv_zcopy)
{
    test_common(&test_rc_mp_xrq::send_rndv_zcopy, 1, 1, false);
}

UCS_TEST_P(test_rc_mp_xrq, rndv_request)
{
    test_common(&test_rc_mp_xrq::send_rndv_request, 1, 1, false);
}

// !! Do not instantiate test_rc_mp_xrq now, until MP XRQ support is upstreamed
//_UCT_INSTANTIATE_TEST_CASE(test_rc_mp_xrq, rc_mlx5)
#endif


#if ENABLE_STATS

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


UCS_TEST_P(test_rc_flow_control_stats, general)
{
    test_general(5, 2, 1);
}

UCS_TEST_P(test_rc_flow_control_stats, soft_request)
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

UCT_RC_INSTANTIATE_TEST_CASE(test_rc_flow_control_stats)

#endif
