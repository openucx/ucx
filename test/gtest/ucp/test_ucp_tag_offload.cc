/**
* Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "test_ucp_tag.h"
#include "ucp_datatype.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/tag/tag_match.h>
}

#define UCP_INSTANTIATE_TAG_OFFLOAD_TEST_CASE(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dcx, "dc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rcx, "rc_x")

class test_ucp_tag_offload : public test_ucp_tag {
public:
    test_ucp_tag_offload() {
        // TODO: test offload and offload MP as different variants
        enable_tag_mp_offload();
    }

    void init()
    {
        test_ucp_tag::init();
        check_offload_support(true);
    }

    request* recv_nb_and_check(void *buffer, size_t count, ucp_datatype_t dt,
                               ucp_tag_t tag, ucp_tag_t tag_mask)
    {
        request *req = recv_nb(buffer, count, dt, tag, tag_mask);
        EXPECT_TRUE(!UCS_PTR_IS_ERR(req));
        EXPECT_TRUE(req != NULL);
        return req;
    }

    request* recv_nb_exp(void *buffer, size_t count, ucp_datatype_t dt,
                         ucp_tag_t tag, ucp_tag_t tag_mask)
    {
        request *req1 = recv_nb_and_check(buffer, count, DATATYPE, tag,
                                          UCP_TAG_MASK_FULL);

        // Post and cancel another receive to make sure the first one was offloaded
        size_t size = receiver().worker()->context->config.ext.tm_thresh + 1;
        std::vector<char> tbuf(size, 0);
        request *req2 = recv_nb_and_check(&tbuf[0], size, DATATYPE, tag,
                                          UCP_TAG_MASK_FULL);
        req_cancel(receiver(), req2);

        return req1;
    }

    void send_recv(entity &se, ucp_tag_t tag, size_t length,
                   ucp_datatype_t rx_dt = DATATYPE)
    {
        std::vector<uint8_t> sendbuf(length);
        std::vector<uint8_t> recvbuf(length);

        request *rreq = recv_nb_exp(&recvbuf[0], length, rx_dt, tag,
                                    UCP_TAG_MASK_FULL);

        request *sreq = (request*)ucp_tag_send_nb(se.ep(), &sendbuf[0], length,
                                                  DATATYPE, tag, send_callback);
        if (UCS_PTR_IS_ERR(sreq)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(sreq));
        } else if (sreq != NULL) {
            wait(sreq);
            request_free(sreq);
        }

        wait(rreq);
        request_free(rreq);
    }

    void activate_offload(entity &se, ucp_tag_t tag = 0x11)
    {
        send_recv(se, tag, receiver().worker()->context->config.ext.tm_thresh);
    }

    void req_cancel(entity &e, request *req)
    {
        ucp_request_cancel(e.worker(), req);
        wait(req);
        request_free(req);
    }
};

UCS_TEST_P(test_ucp_tag_offload, post_after_cancel)
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag      = 0x11;
    std::vector<char> recvbuf(2048, 0);

    activate_offload(sender());

    request *req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                     tag, UCP_TAG_MASK_FULL);

    EXPECT_EQ(1u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);
    EXPECT_EQ(0u, receiver().worker()->tm.expected.sw_all_count);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag,
                            UCP_TAG_MASK_FULL);

    EXPECT_EQ(0u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);
}

UCS_TEST_P(test_ucp_tag_offload, post_after_comp)
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag      = 0x11;
    std::vector<char> recvbuf(2048, 0);

    activate_offload(sender());

    request *req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                     tag, UCP_TAG_MASK_FULL);

    EXPECT_EQ(1u, receiver().worker()->tm.expected.sw_all_count);

    send_b(&small_val, sizeof(small_val), DATATYPE, tag);
    wait(req);
    request_free(req);
    EXPECT_EQ(0u, receiver().worker()->tm.expected.sw_all_count);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag,
                            UCP_TAG_MASK_FULL);

    EXPECT_EQ(0u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req);
}

UCS_TEST_P(test_ucp_tag_offload, post_wild)
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag1     = 0x11; // these two tags should go to different
    ucp_tag_t tag2     = 0x13; // hash buckets in the TM expected queue
    std::vector<char> recvbuf(2048, 0);

    activate_offload(sender());

    request *req1 = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                      tag1, 0);
    EXPECT_EQ(1u, receiver().worker()->tm.expected.sw_all_count);

    request *req2 = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag2,
                                      UCP_TAG_MASK_FULL);
    // Second request should not be posted as well. Even though it has another
    // tag, the first request is a wildcard, which needs to be handled in SW,
    // so it blocks all other requests
    EXPECT_EQ(2u, receiver().worker()->tm.expected.sw_all_count);
    req_cancel(receiver(), req1);
    req_cancel(receiver(), req2);
}

UCS_TEST_P(test_ucp_tag_offload, post_dif_buckets)
{
    uint64_t small_val = 0xFAFA;
    ucp_tag_t tag1     = 0x11; // these two tags should go to different
    ucp_tag_t tag2     = 0x13; // hash buckets in the TM expected queue
    std::vector<request*> reqs;
    request *req;

    std::vector<char> recvbuf(2048, 0);

    activate_offload(sender());

    req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE, tag1,
                            UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag1,
                            UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    // The first request was not offloaded due to small size and the second
    // is blocked by the first one.
    EXPECT_EQ(2u, receiver().worker()->tm.expected.sw_all_count);

    req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE, tag2,
                            UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    // Check that another request with different tag is offloaded.
    EXPECT_EQ(2u, receiver().worker()->tm.expected.sw_all_count);

    for (std::vector<request*>::const_iterator iter = reqs.begin();
         iter != reqs.end(); ++iter) {
        req_cancel(receiver(), *iter);
    }
}

UCS_TEST_P(test_ucp_tag_offload, force_thresh_basic, "TM_FORCE_THRESH=4k",
                                                     "TM_THRESH=1k")
{
    uint64_t small_val      = 0xFAFA;
    const size_t big_size   = 5000;
    int num_reqs            = 8;
    int tag                 = 0x11;
    std::vector<request*> reqs;
    request *req;

    activate_offload(sender());

    for (int i = 0; i < num_reqs - 1; ++i) {
        req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                tag, UCP_TAG_MASK_FULL);
        reqs.push_back(req);
    }

    // No requests should be posted to the transport, because their sizes less
    // than TM_THRESH
    EXPECT_EQ((unsigned)(num_reqs - 1), receiver().worker()->tm.expected.sw_all_count);

    std::vector<char> recvbuf_big(big_size, 0);

    req = recv_nb(&recvbuf_big[0], recvbuf_big.size(), DATATYPE, tag,
                  UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    // Now, all requests should be posted to the transport, because receive
    // buffer bigger than FORCE_THRESH has been posted
    EXPECT_EQ((unsigned)0, receiver().worker()->tm.expected.sw_all_count);

    std::vector<request*>::const_iterator iter;
    for (iter = reqs.begin(); iter != reqs.end(); ++iter) {
        req_cancel(receiver(), *iter);
    }
}

UCS_TEST_P(test_ucp_tag_offload, force_thresh_blocked, "TM_FORCE_THRESH=4k",
                                                       "TM_THRESH=1k")
{
    uint64_t small_val      = 0xFAFA;
    const size_t big_size   = 5000;
    int num_reqs            = 8;
    int tag                 = 0x11;
    std::vector<request*> reqs;
    request *req;
    int i;

    activate_offload(sender());

    for (i = 0; i < num_reqs - 3; ++i) {
        req = recv_nb_and_check(&small_val, sizeof(small_val), DATATYPE,
                                tag, UCP_TAG_MASK_FULL);
        reqs.push_back(req);
    }

    // Add request with noncontig dt
    std::vector<char> buf(64, 0);
    ucp::data_type_desc_t dt_desc(DATATYPE_IOV, buf.data(), buf.size(), 1);
    req = recv_nb_and_check(dt_desc.buf(), dt_desc.count(), dt_desc.dt(),
                            tag, UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    // Add request with wildcard tag
    req = recv_nb(&small_val, sizeof(small_val), DATATYPE, tag, 0);
    reqs.push_back(req);

    std::vector<char> recvbuf_big(big_size, 0);
    // Check that offload is not forced while there are uncompleted blocking
    // SW requests with the same tag
    for (i = 0; i < 2; ++i) {
        req = recv_nb_and_check(&recvbuf_big[0], recvbuf_big.size(), DATATYPE, tag,
                                UCP_TAG_MASK_FULL);
        EXPECT_EQ((unsigned)(num_reqs - i), receiver().worker()->tm.expected.sw_all_count);
        req_cancel(receiver(), req);

        req_cancel(receiver(), reqs.back());
        reqs.pop_back();
    }

    req = recv_nb(&recvbuf_big[0], recvbuf_big.size(), DATATYPE, tag,
                  UCP_TAG_MASK_FULL);
    reqs.push_back(req);

    // Now, all requests should be posted to the transport, because receive
    // buffer bigger than FORCE_THRESH has been posted
    EXPECT_EQ((unsigned)0, receiver().worker()->tm.expected.sw_all_count);

    std::vector<request*>::const_iterator iter;
    for (iter = reqs.begin(); iter != reqs.end(); ++iter) {
        req_cancel(receiver(), *iter);
    }
}

// Check that worker will not try to connect tag offload capable iface with
// the peer which does not support tag offload (e.g CX-5 and CX-4). In this
// case connection attempt should fail (due to peer unreachable) or some other
// transport should be selected (if available). Otherwise connect can hang,
// because some transports (e.g. rcx) have different ep address type for
// interfaces which support tag_offload.
UCS_TEST_P(test_ucp_tag_offload, connect)
{
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_ENABLE", "n"));

    entity *e = create_entity(true);
    // Should be:
    // - either complete ok
    // - or force skipping the test (because peer is unreachable)
    e->connect(&receiver(), get_ep_params());
}

// Send small chunk of data to be scattered to CQE on the receiver. Post bigger
// chunk of memory for receive operation, so it would be posted to the HW.
UCS_TEST_P(test_ucp_tag_offload, eager_send_less, "RNDV_THRESH=inf",
           "TM_THRESH=0", "TM_MAX_BB_SIZE=0")
{
    activate_offload(sender());

    uint8_t              send_data = 0;
    size_t               length    = 4 * UCS_KBYTE;
    ucp_tag_t            tag       = 0x11;
    std::vector<uint8_t> recvbuf(length);

    request *rreq = recv_nb_exp(&recvbuf[0], length, ucp_dt_make_contig(1), tag,
                                UCP_TAG_MASK_FULL);

    request *sreq = (request*)ucp_tag_send_nb(sender().ep(), &send_data,
                                              sizeof(send_data),
                                              ucp_dt_make_contig(1), tag,
                                              send_callback);
    if (UCS_PTR_IS_ERR(sreq)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(sreq));
    } else if (sreq != NULL) {
        request_wait(sreq);
    }

    request_wait(rreq);
}

UCS_TEST_P(test_ucp_tag_offload, small_rndv, "RNDV_THRESH=0", "TM_THRESH=0")
{
    activate_offload(sender());
    send_recv(sender(), 0x11ul, 0ul);
    send_recv(sender(), 0x11ul, 1ul);
}

UCS_TEST_P(test_ucp_tag_offload, small_sw_rndv, "RNDV_THRESH=0", "TM_THRESH=0",
                                                "TM_SW_RNDV=y")
{
    activate_offload(sender());
    send_recv(sender(), 0x11ul, 0ul);
    send_recv(sender(), 0x11ul, 1ul);
}

UCS_TEST_P(test_ucp_tag_offload, sw_rndv_rx_generic, "RNDV_THRESH=0",
                                                     "TM_THRESH=0",
                                                     "TM_SW_RNDV=y")
{
    activate_offload(sender());

    ucp_datatype_t ucp_dt;
    ASSERT_UCS_OK(ucp_dt_create_generic(&ucp::test_dt_copy_ops, NULL,
                                        &ucp_dt));

    send_recv(sender(), 0x11ul, 4 * UCS_KBYTE, ucp_dt);

    ucp_dt_destroy(ucp_dt);
}

UCS_TEST_P(test_ucp_tag_offload, eager_multi_probe,
           "RNDV_THRESH=inf", "TM_THRESH=0")
{
    activate_offload(sender());

    size_t length = ucp_ep_config(sender().ep())->tag.rndv.am_thresh.remote - 1;
    ucp_tag_t tag = 0x11;
    std::vector<uint8_t> sendbuf(length);

    ucs_status_ptr_t sreq = ucp_tag_send_nb(sender().ep(), sendbuf.data(),
                                          sendbuf.size(), ucp_dt_make_contig(1),
                                          tag, send_callback);

    ucp_tag_recv_info_t info;
    ucp_tag_message_h msg = NULL;
    ucs_time_t deadline = ucs::get_deadline();
    while ((msg == NULL) && (ucs_get_time() < deadline)) {
        progress();
        msg = ucp_tag_probe_nb(receiver().worker(), tag, 0xffff, 1, &info);
    }
    EXPECT_EQ(length, info.length);

    std::vector<uint8_t> recvbuf(length);
    ucs_status_ptr_t rreq = ucp_tag_msg_recv_nb(receiver().worker(),
                                                &recvbuf[0], length,
                                                ucp_dt_make_contig(1),
                                                msg, recv_callback);
    request_wait(sreq);
    request_wait(rreq);
}

// Test that message is received correctly if the corresponging receive
// operation was posted when the first (but not all) fragments arrived. This is
// to ensure that the following sequence does not happen:
// 1. First fragment arrives and is not added to the unexp queue
// 2. Receive operation matching this messages is invoked and the message is not
//    found in the unexp queue
// 3. When all fragments arrive and the message is added to the unexp queue, it
//    may be matched by another receive operation causing ordering issues.
//    Or, if it is the only message to receive (like it is in this test), the
//    receive operation (invoked at step 2) will never complete.
UCS_TEST_P(test_ucp_tag_offload, eager_multi_recv,
           "RNDV_THRESH=inf", "TM_THRESH=0")
{
    activate_offload(sender());

    size_t length = ucp_ep_config(sender().ep())->tag.rndv.am_thresh.remote - 1;
    const ucp_tag_t tag = 0x11;
    std::vector<uint8_t> sendbuf(length);

    ucp_request_param_t param = {};
    ucs_status_ptr_t sreq     = ucp_tag_send_nbx(sender().ep(), sendbuf.data(),
                                                 sendbuf.size(), tag, &param);

    // Tweak progress several times to make sure the first fragment
    // (but not all!) arrives
    for (int i = 0; i < 3; ++i) {
        progress();
    }

    std::vector<uint8_t> recvbuf(length);
    ucs_status_ptr_t rreq = ucp_tag_recv_nbx(receiver().worker(), recvbuf.data(),
                                             length, tag, 0xffff, &param);
    request_wait(sreq);
    request_wait(rreq);
}

UCP_INSTANTIATE_TAG_OFFLOAD_TEST_CASE(test_ucp_tag_offload)


class test_ucp_tag_offload_multi : public test_ucp_tag_offload {
public:

    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        ucp_params_t params    = test_ucp_tag::get_ctx_params();
        params.field_mask     |= UCP_PARAM_FIELD_TAG_SENDER_MASK;
        params.tag_sender_mask = TAG_SENDER;
        add_variant(variants, params);
    }

    void init()
    {
        // The test checks that increase of active ifaces is handled
        // correctly. It needs to start with a single active iface, therefore
        // disable multi-rail.
        modify_config("MAX_EAGER_LANES", "1");
        modify_config("MAX_RNDV_LANES",  "1");

        test_ucp_tag_offload::init();

        // TODO: add more tls which support tag offloading
        std::vector<std::string> tls;
        tls.push_back("dc_x");
        tls.push_back("rc_x");
        ucp_test_param params = GetParam();

        // Create new entity and add to to the end of vector
        // (thus it will be receiver without any connections)
        create_entity(false);
        for (std::vector<std::string>::const_iterator i = tls.begin();
             i != tls.end(); ++i) {
            params.transports.clear();
            params.transports.push_back(*i);
            create_entity(true, params);
            sender().connect(&receiver(), get_ep_params());
            check_offload_support(true);
        }
    }

    ucp_tag_t make_tag(entity &e, ucp_tag_t t)
    {
        uint64_t i;

        for (i = 0; i < m_entities.size(); ++i) {
             if (&m_entities.at(i) == &e) {
                 break;
             }
        }
        return (i << 48) | t;
    }

    void activate_offload_hashing(entity &se, ucp_tag_t tag)
    {
        se.connect(&receiver(), get_ep_params());
        // Need to send twice:
        // 1. to ensure that wireup's UCT iface has been closed and it is not
        //    considered for num_active_iface on worker (message has to be less
        //    than `UCX_TM_THRESH` value) + UCP workers have to be flushed prior
        //    to ensure that UCT ifaces were deactivated at the end of auxiliary
        //    UCT EP discarding
        // 2. to activate tag ofload
        //    (num_active_ifaces on worker is increased when any message is
        //    received on any iface. Tag hashing is done when we have more than
        //    1 active ifaces and message has to be greater than `UCX_TM_THRESH`
        //    value)
        flush_workers();
        send_recv(se, tag, 8);
        send_recv(se, tag, 2048);
    }

    void post_recv_and_check(entity &e, unsigned sw_count, ucp_tag_t tag,
                             ucp_tag_t tag_mask)
    {
        std::vector<char> recvbuf(2048, 0);
        request *req = recv_nb_and_check(&recvbuf, recvbuf.size(), DATATYPE,
                                         make_tag(e, tag), UCP_TAG_MASK_FULL);

        EXPECT_EQ(sw_count, receiver().worker()->tm.expected.sw_all_count);
        req_cancel(receiver(), req);
    }


protected:
    static const uint64_t TAG_SENDER = 0xFFFFFFFFFFFF0000;
};


UCS_TEST_P(test_ucp_tag_offload_multi, recv_from_multi)
{
    ucp_tag_t tag = 0x11;

    // Activate first offload iface. Tag hashing is not done yet, since we
    // have only one active iface so far.
    activate_offload_hashing(e(0), make_tag(e(0), tag));
    EXPECT_EQ(0u, kh_size(&receiver().worker()->tm.offload.tag_hash));

    // Activate second offload iface. The tag has been added to the hash.
    // From now requests will be offloaded only for those tags which are
    // in the hash.
    activate_offload_hashing(e(1), make_tag(e(1), tag));
    EXPECT_EQ(1u, kh_size(&receiver().worker()->tm.offload.tag_hash));

    // Need to send a message on the first iface again, for its 'tag_sender'
    // part of the tag to be added to the hash.
    send_recv(e(0), make_tag(e(0), tag), 2048);
    EXPECT_EQ(2u, kh_size(&receiver().worker()->tm.offload.tag_hash));

    // Now requests from first two senders should be always offloaded regardless
    // of the tag value. Tag does not matter, because hashing is done with
    // 'tag & tag_sender_mask' as a key.
    for (int i = 0; i < 2; ++i) {
        post_recv_and_check(e(i), 0u, tag + i, UCP_TAG_MASK_FULL);
    }

    // This request should not be offloaded, because it is sent by the new
    // sender and its 'tag_sender_mask' is not added to the hash yet.
    post_recv_and_check(e(2), 1u, tag, UCP_TAG_MASK_FULL);

    activate_offload_hashing(e(2), make_tag(e(2), tag));
    EXPECT_EQ(3u, kh_size(&receiver().worker()->tm.offload.tag_hash));

    // Check that this sender was added as well
    post_recv_and_check(e(2), 0u, tag + 1, UCP_TAG_MASK_FULL);
}

// Do not include SM transports, because they would be selected for tag matching.
// And since they do not support TM offload, this test would be skipped.
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_tag_offload_multi, all_rcdc, "rc,dc")


class test_ucp_tag_offload_selection : public test_ucp_tag_offload {
public:
    test_ucp_tag_offload_selection() {
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_ENABLE", "y"));
    }

    static uct_device_type_t get_dev_type(ucp_ep_h ep, ucp_rsc_index_t idx) {
        return ep->worker->context->tl_rscs[idx].tl_rsc.dev_type;
    }

    static bool lane_shm_or_self(ucp_ep_h ep, ucp_rsc_index_t idx) {
        uct_device_type_t dev_type = get_dev_type(ep, idx);
        return (dev_type == UCT_DEVICE_TYPE_SHM) || (dev_type == UCT_DEVICE_TYPE_SELF);
    }
};

UCS_TEST_P(test_ucp_tag_offload_selection, tag_lane)
{
    ucp_ep_h ep          = sender().ep();
    bool has_tag_offload = false;
    bool has_shm_or_self = false;

    for (ucp_rsc_index_t idx = 0; idx < sender().ucph()->num_tls; ++idx) {
        if (lane_shm_or_self(ep, idx)) {
            has_shm_or_self = true;
        }

        uct_iface_attr_t *attr = ucp_worker_iface_get_attr(sender().worker(), idx);
        if (attr->cap.flags & UCT_IFACE_FLAG_TAG_EAGER_BCOPY) {
            // We do not have transports with partial tag offload support
            EXPECT_TRUE(attr->cap.flags & UCT_IFACE_FLAG_TAG_RNDV_ZCOPY);
            has_tag_offload = true;
        }
    }

    ucp_ep_config_t *ep_config = ucp_ep_config(ep);

    if (has_tag_offload && !has_shm_or_self) {
        EXPECT_TRUE(ucp_ep_config_key_has_tag_lane(&ep_config->key));
        EXPECT_EQ(ep_config->key.tag_lane, ep_config->tag.lane);
    } else {
        // If shm or self transports exist they would be used for tag matching
        // rather than network offload
        EXPECT_FALSE(ucp_ep_config_key_has_tag_lane(&ep_config->key));
        EXPECT_EQ(ep_config->key.am_lane, ep_config->tag.lane);
    }
}

UCP_INSTANTIATE_TAG_OFFLOAD_TEST_CASE(test_ucp_tag_offload_selection);
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_tag_offload_selection, self_rcx,
                              "self,rc_x");


class test_ucp_tag_offload_gpu : public test_ucp_tag_offload {
public:
    test_ucp_tag_offload_gpu() {
        modify_config("RNDV_THRESH", "1024");
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_memtypes(variants, test_ucp_tag::get_test_variants,
                             UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                             UCS_BIT(UCS_MEMORY_TYPE_ROCM));
    }

protected:
    ucs_memory_type_t mem_type() const {
        return static_cast<ucs_memory_type_t>(get_variant_value());
    }
};

// Test that expected SW RNDV request is handled properly when receive buffer
// is allocated on GPU memory.
UCS_TEST_P(test_ucp_tag_offload_gpu, sw_rndv_to_gpu_mem, "TM_SW_RNDV=y")
{
    activate_offload(sender());

    size_t size   = 2048;
    ucp_tag_t tag = 0xCAFEBABEul;
    // Test will be skipped here if GPU mem is not supported
    mem_buffer rbuf(size, mem_type());
    request *rreq = recv_nb_exp(rbuf.ptr(), size, DATATYPE, tag,
                                UCP_TAG_MASK_FULL);

    std::vector<uint8_t> sendbuf(size); // can send from any memory
    request *sreq = (request*)ucp_tag_send_nb(sender().ep(), &sendbuf[0],
                                              size, DATATYPE, tag,
                                              send_callback);
    wait_and_validate(rreq);
    wait_and_validate(sreq);
}

// Test that small buffers wich can be scattered to CQE are not posted to the
// HW. Otherwise it may segfault, while copying data from CQE to the
// (potentially) GPU buffer.
UCS_TEST_P(test_ucp_tag_offload_gpu, rx_scatter_to_cqe, "TM_THRESH=1")
{
    activate_offload(sender());

    size_t size   = 8;
    ucp_tag_t tag = 0xCAFEBABEul;
    // Test will be skipped here if GPU mem is not supported
    mem_buffer rbuf(size, mem_type());
    request *rreq = recv_nb_exp(rbuf.ptr(), size, DATATYPE, tag,
                                UCP_TAG_MASK_FULL);
    uint64_t sbuf = 0ul;
    request *sreq = (request*)ucp_tag_send_nb(sender().ep(), &sbuf, sizeof(sbuf),
                                              DATATYPE, tag, send_callback);
    wait_and_validate(rreq);
    wait_and_validate(sreq);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_tag_offload_gpu, rc_dc_gpu,
                                        "dc_x,rc_x")

class test_ucp_tag_offload_status : public test_ucp_tag {
public:
    test_ucp_tag_offload_status() {
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_ENABLE", "y"));
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        // Do not pass UCP_FEATURE_TAG feature to check that UCT will not
        // initialize tag offload infrastructure in this case.
        add_variant(variants, UCP_FEATURE_RMA);
    }
};

UCS_TEST_P(test_ucp_tag_offload_status, check_offload_status)
{
    for (ucp_rsc_index_t i = 0; i < sender().ucph()->num_tls; ++i) {
        EXPECT_FALSE(ucp_worker_iface_get_attr(sender().worker(), i)->cap.flags &
                     (UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                      UCT_IFACE_FLAG_TAG_RNDV_ZCOPY));
    }
}

UCP_INSTANTIATE_TAG_OFFLOAD_TEST_CASE(test_ucp_tag_offload_status)

#ifdef ENABLE_STATS

class test_ucp_tag_offload_stats : public test_ucp_tag_offload_multi {
public:

    void init()
    {
        stats_activate();
        test_ucp_tag_offload::init(); // No need for multi::init()
    }

    void cleanup()
    {
        test_ucp_tag_offload::cleanup();
        stats_restore();
    }

    request* recv_nb_exp(void *buffer, size_t count, ucp_datatype_t dt,
                         ucp_tag_t tag, ucp_tag_t tag_mask)
    {
        request *req1 = recv_nb_and_check(buffer, count, DATATYPE, tag,
                                          UCP_TAG_MASK_FULL);

        // Post and cancel another receive to make sure the first one was offloaded
        size_t size = receiver().worker()->context->config.ext.tm_thresh + 1;
        std::vector<char> tbuf(size, 0);
        request *req2 = recv_nb_and_check(&tbuf[0], size, DATATYPE, tag,
                                          UCP_TAG_MASK_FULL);
        req_cancel(receiver(), req2);

        return req1;
    }

    ucs_stats_node_t* worker_offload_stats(entity &e)
    {
        return e.worker()->tm_offload_stats;
    }

    void validate_offload_counter(uint64_t rx_cntr, uint64_t val)
    {
        uint64_t cnt;
        cnt = UCS_STATS_GET_COUNTER(worker_offload_stats(receiver()), rx_cntr);
        EXPECT_EQ(val, cnt);
    }

    void wait_counter(ucs_stats_node_t *stats, uint64_t cntr,
                      double timeout = ucs::test_timeout_in_sec)
    {
        ucs_time_t deadline = ucs::get_deadline(timeout);
        uint64_t   v;

        do {
            short_progress_loop();
            v = UCS_STATS_GET_COUNTER(stats, cntr);
        } while ((ucs_get_time() < deadline) && !v);

        EXPECT_EQ(1ul, v);
    }

    void test_send_recv(size_t count, bool send_iov, uint64_t cntr)
    {
        ucp_tag_t tag = 0x11;

        std::vector<char> sbuf(count, 0);
        std::vector<char> rbuf(count, 0);
        request *req = recv_nb_exp(rbuf.data(), rbuf.size(), DATATYPE, tag,
                                   UCP_TAG_MASK_FULL);

        if (send_iov) {
            ucp::data_type_desc_t dt_desc(DATATYPE_IOV, sbuf.data(),
                                          sbuf.size(), 1);
            send_b(dt_desc.buf(), dt_desc.count(), dt_desc.dt(), tag);
        } else {
            send_b(sbuf.data(), sbuf.size(), DATATYPE, tag);
        }
        wait(req);
        request_free(req);

        validate_offload_counter(cntr, 1ul);
    }
};

UCS_TEST_P(test_ucp_tag_offload_stats, post, "TM_THRESH=1")
{
    uint64_t tag = 0x11;
    uint64_t dummy;

    activate_offload(sender());

    request *rreq = recv_nb(&dummy, sizeof(dummy), DATATYPE, tag,
                            UCP_TAG_MASK_FULL);

    wait_counter(worker_offload_stats(receiver()),
                 UCP_WORKER_STAT_TAG_OFFLOAD_POSTED);

    req_cancel(receiver(), rreq);

    wait_counter(worker_offload_stats(receiver()),
                 UCP_WORKER_STAT_TAG_OFFLOAD_CANCELED);
}

UCS_TEST_P(test_ucp_tag_offload_stats, block, "TM_THRESH=1")
{
    uint64_t tag = 0x11;
    std::vector<char> buf(64, 0);

    activate_offload(sender());

    // Check BLOCK_NON_CONTIG
    ucp::data_type_desc_t dt_desc(DATATYPE_IOV, buf.data(), buf.size(), 1);
    request *rreq = recv_nb_and_check(dt_desc.buf(), dt_desc.count(),
                                      dt_desc.dt(), tag, UCP_TAG_MASK_FULL);

    wait_counter(worker_offload_stats(receiver()),
                 UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_NON_CONTIG);

    req_cancel(receiver(), rreq);

    // Check BLOCK_WILDCARD
    rreq = recv_nb_and_check(buf.data(), buf.size(), DATATYPE, tag, 0);

    wait_counter(worker_offload_stats(receiver()),
                 UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_WILDCARD);

    req_cancel(receiver(), rreq);

    // Check BLOCK_TAG_EXCEED
    std::vector<request*> reqs;
    uint64_t cnt;
    unsigned limit = 1000; // Just a big value to avoid test hang
    do {
        rreq = recv_nb_and_check(buf.data(), buf.size(), DATATYPE, tag,
                                 UCP_TAG_MASK_FULL);
        cnt  = UCS_STATS_GET_COUNTER(worker_offload_stats(receiver()),
                                    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_TAG_EXCEED);
        reqs.push_back(rreq);
    } while (!cnt && (--limit > 0));

    validate_offload_counter(UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_TAG_EXCEED , 1ul);

    for (std::vector<request*>::const_iterator iter = reqs.begin();
         iter != reqs.end(); ++iter) {
        req_cancel(receiver(), *iter);
    }
}

UCS_TEST_P(test_ucp_tag_offload_stats, eager, "RNDV_THRESH=1000", "TM_THRESH=64")
{
    size_t size = 512; // Size smaller than RNDV, but bigger than TM thresh

    // Offload is not activated, so the first message should arrive unexpectedly
    test_send_recv(size, false, UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_EGR);
    test_send_recv(size, false, UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED);
}

UCS_TEST_P(test_ucp_tag_offload_stats, rndv, "RNDV_THRESH=1000")
{
    size_t size = 2048; // Size bigger than RNDV thresh

    // Offload is not activated, so the first message should arrive unexpectedly
    test_send_recv(size, false, UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_RNDV);
    test_send_recv(size, false, UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED);
}

UCS_TEST_P(test_ucp_tag_offload_stats, sw_rndv, "RNDV_THRESH=1000")
{
    size_t size = 2048; // Size bigger than RNDV thresh

    // Offload is not activated, so the first message should arrive unexpectedly
    test_send_recv(size, true, UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_SW_RNDV);
    test_send_recv(size, true, UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED_SW_RNDV);
}

UCS_TEST_P(test_ucp_tag_offload_stats, force_sw_rndv, "TM_SW_RNDV=y",
                                                      "RNDV_THRESH=1000")
{
    size_t size = 2048; // Size bigger than RNDV thresh

    // Offload is not activated, so the first message should arrive unexpectedly
    test_send_recv(size, false, UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_SW_RNDV);
    test_send_recv(size, false, UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED_SW_RNDV);
}


UCP_INSTANTIATE_TAG_OFFLOAD_TEST_CASE(test_ucp_tag_offload_stats)


class test_ucp_tag_offload_stats_gpu : public test_ucp_tag_offload_stats {
public:
    test_ucp_tag_offload_stats_gpu() {
        m_env.push_back(new ucs::scoped_setenv("UCX_IB_GPU_DIRECT_RDMA", "n"));
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_memtypes(variants,
                             test_ucp_tag_offload_stats::get_test_variants,
                             UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                             UCS_BIT(UCS_MEMORY_TYPE_ROCM));
    }

protected:
    ucs_memory_type_t mem_type() const {
        return static_cast<ucs_memory_type_t>(get_variant_value());
    }
};

UCS_TEST_P(test_ucp_tag_offload_stats_gpu, block_gpu_no_gpu_direct,
           "TM_THRESH=1")
{
    activate_offload(sender());

    size_t size   = 2048;
    // Test will be skipped here if GPU mem is not supported
    mem_buffer rbuf(size, mem_type());
    request *rreq = recv_nb_and_check(rbuf.ptr(), size, DATATYPE, 0x11,
                                      UCP_TAG_MASK_FULL);

    wait_counter(worker_offload_stats(receiver()),
                 UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_MEM_REG);

    validate_offload_counter(UCP_WORKER_STAT_TAG_OFFLOAD_POSTED, 0ul);

    req_cancel(receiver(), rreq);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_tag_offload_stats_gpu,
                                        rc_dc_gpu, "dc_x,rc_x")

#endif
