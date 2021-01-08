/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "ucp_test.h"
#include "common/test.h"
#include "ucp/ucp_test.h"

#include <algorithm>
#include <set>

extern "C" {
#include <ucp/wireup/address.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/sys/math.h>
}

class test_ucp_wireup : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants,
                                  uint64_t features, bool test_all = false);

protected:
    enum {
        TEST_RMA     = UCS_BIT(0),
        TEST_TAG     = UCS_BIT(1),
        TEST_STREAM  = UCS_BIT(2),
        UNIFIED_MODE = UCS_BIT(3),
        TEST_AMO     = UCS_BIT(4)
    };

    typedef uint64_t               elem_type;
    typedef std::vector<elem_type> vec_type;

    static const size_t BUFFER_LENGTH    = 16384;
    static const ucp_datatype_t DT_U64 = ucp_dt_make_contig(sizeof(elem_type));
    static const uint64_t TAG          = 0xdeadbeef;
    static const elem_type SEND_DATA   = 0xdeadbeef12121212ull;

    virtual void init();
    virtual void cleanup();

    void send_nb(ucp_ep_h ep, size_t length, int repeat, std::vector<void*>& reqs,
                 uint64_t send_data = SEND_DATA);

    void send_b(ucp_ep_h ep, size_t length, int repeat,
                uint64_t send_data = SEND_DATA);

    void recv_b(ucp_worker_h worker, ucp_ep_h ep, size_t length, int repeat,
                uint64_t recv_data = SEND_DATA);

    void send_recv(ucp_ep_h send_ep, ucp_worker_h recv_worker, ucp_ep_h recv_ep,
                   size_t vecsize, int repeat);

    void waitall(std::vector<void*> reqs);

    void disconnect(ucp_ep_h ep);

    void disconnect(ucp_test::entity &e);

    static void close_completion(void *request, ucs_status_t status,
                                 void *user_data);

    static void send_completion(void *request, ucs_status_t status);

    static void tag_recv_completion(void *request, ucs_status_t status,
                                    ucp_tag_recv_info_t *info);

    void rkeys_cleanup();

    void memhs_cleanup();

    void clear_recv_data();

    void fill_send_data();

    ucp_rkey_h get_rkey(ucp_ep_h ep, ucp_mem_h memh);

    bool ep_iface_has_caps(const entity& e, const std::string& tl,
                           uint64_t caps);

protected:
    vec_type                               m_send_data;
    vec_type                               m_recv_data;
    ucs::handle<ucp_mem_h, ucp_context_h>  m_memh_sender;
    ucs::handle<ucp_mem_h, ucp_context_h>  m_memh_receiver;
    std::vector< ucs::handle<ucp_rkey_h> > m_rkeys;

private:
    static void stream_recv_completion(void *request, ucs_status_t status,
                                       size_t length);

    static void unmap_memh(ucp_mem_h memh, ucp_context_h context);
};

void test_ucp_wireup::get_test_variants(std::vector<ucp_test_variant>& variants,
                                        uint64_t features, bool test_all)
{
    std::vector<ucp_test_param> result;

    if (features & UCP_FEATURE_RMA) {
        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_RMA, "rma");
        add_variant_with_value(variants, UCP_FEATURE_RMA,
                               TEST_RMA | UNIFIED_MODE, "rma,unified");
    }

    if (features & UCP_FEATURE_TAG) {
        add_variant_with_value(variants, UCP_FEATURE_TAG, TEST_TAG, "tag");
        add_variant_with_value(variants, UCP_FEATURE_TAG,
                               TEST_TAG | UNIFIED_MODE, "tag,unified");
    }

    if (features & UCP_FEATURE_STREAM) {
        add_variant_with_value(variants, UCP_FEATURE_STREAM, TEST_STREAM, "stream");
        add_variant_with_value(variants, UCP_FEATURE_STREAM,
                               TEST_STREAM | UNIFIED_MODE, "stream,unified");
    }

    if (features & (UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64)) {
        add_variant_with_value(variants, UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64,
                               TEST_AMO, "amo");
    }

    if (test_all) {
        uint64_t all_flags = (TEST_TAG | TEST_RMA | TEST_STREAM);
        add_variant_with_value(variants, features, all_flags, "all");
        add_variant_with_value(variants, features, all_flags | UNIFIED_MODE,
                               "all,unified");
    }
}

void test_ucp_wireup::unmap_memh(ucp_mem_h memh, ucp_context_h context)
{
    ucs_status_t status = ucp_mem_unmap(context, memh);
    if (status != UCS_OK) {
        ucs_warn("failed to unmap memory: %s", ucs_status_string(status));
    }
}

void test_ucp_wireup::init()
{
    if (get_variant_value() & UNIFIED_MODE) {
        modify_config("UNIFIED_MODE",  "y");
    }

    ucp_test::init();

    m_send_data.resize(BUFFER_LENGTH, 0);
    m_recv_data.resize(BUFFER_LENGTH, 0);

    if (get_variant_value() & (TEST_RMA | TEST_AMO)) {
        ucs_status_t status;
        ucp_mem_map_params_t params;
        ucp_mem_h memh;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = &m_recv_data[0];
        params.length     = m_recv_data.size() * sizeof(m_recv_data[0]);
        params.flags      = 0;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);
        m_memh_sender.reset(memh, unmap_memh, sender().ucph());

        status = ucp_mem_map(receiver().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);
        m_memh_receiver.reset(memh, unmap_memh, receiver().ucph());
    }
}

ucp_rkey_h test_ucp_wireup::get_rkey(ucp_ep_h ep, ucp_mem_h memh)
{
    void *rkey_buffer;
    size_t rkey_size;
    ucs_status_t status;
    ucp_rkey_h rkey;

    if (memh == m_memh_receiver) {
        status = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_size);
    } else if (memh == m_memh_sender) {
        status = ucp_rkey_pack(sender().ucph(), memh, &rkey_buffer, &rkey_size);
    } else {
        status = UCS_ERR_INVALID_PARAM;
    }
    ASSERT_UCS_OK(status);

    status = ucp_ep_rkey_unpack(ep, rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);

    return rkey;
}

void test_ucp_wireup::rkeys_cleanup() {
    m_rkeys.clear();
}

void test_ucp_wireup::memhs_cleanup() {
    m_memh_sender.reset();
    m_memh_receiver.reset();
}

void test_ucp_wireup::cleanup() {
    rkeys_cleanup();
    memhs_cleanup();
    ucp_test::cleanup();
}

void test_ucp_wireup::clear_recv_data() {
    std::fill(m_recv_data.begin(), m_recv_data.end(), 0);
}

void test_ucp_wireup::send_nb(ucp_ep_h ep, size_t length, int repeat,
                              std::vector<void*>& reqs, uint64_t send_data)
{
    if (get_variant_value() & TEST_TAG) {
        std::fill(m_send_data.begin(), m_send_data.end(), send_data);
        for (int i = 0; i < repeat; ++i) {
            void *req = ucp_tag_send_nb(ep, &m_send_data[0], length,
                                        DT_U64, TAG, send_completion);
            if (UCS_PTR_IS_PTR(req)) {
                reqs.push_back(req);
            } else {
                ASSERT_UCS_OK(UCS_PTR_STATUS(req));
            }
        }
    } else if (get_variant_value() & TEST_STREAM) {
        std::fill(m_send_data.begin(), m_send_data.end(), send_data);
        for (int i = 0; i < repeat; ++i) {
            void *req = ucp_stream_send_nb(ep, &m_send_data[0], length, DT_U64,
                                           send_completion, 0);
            if (UCS_PTR_IS_PTR(req)) {
                reqs.push_back(req);
            } else {
                ASSERT_UCS_OK(UCS_PTR_STATUS(req));
            }
        }
    } else if (get_variant_value() & TEST_RMA) {
        clear_recv_data();

        ucp_mem_h memh  = (sender().ucph() == ep->worker->context) ?
                            m_memh_receiver : m_memh_sender;
        ucp_rkey_h rkey = get_rkey(ep, memh);

        m_rkeys.push_back(ucs::handle<ucp_rkey_h>(rkey, ucp_rkey_destroy));

        for (int i = 0; i < repeat; ++i) {
            std::fill(m_send_data.begin(), m_send_data.end(), send_data + i);
            void *req = ucp_put_nb(ep, &m_send_data[0],
                                   m_send_data.size() * sizeof(m_send_data[0]),
                                   (uintptr_t)&m_recv_data[0], rkey,
                                   send_completion);
            if (UCS_PTR_IS_PTR(req)) {
                reqs.push_back(req);
            } else {
                ASSERT_UCS_OK(UCS_PTR_STATUS(req));
            }
        }
    }
}

void test_ucp_wireup::send_b(ucp_ep_h ep, size_t length, int repeat,
                             uint64_t send_data)
{
    std::vector<void*> reqs;
    send_nb(ep, length, repeat, reqs, send_data);
    waitall(reqs);
}

void test_ucp_wireup::recv_b(ucp_worker_h worker, ucp_ep_h ep, size_t length,
                             int repeat, uint64_t recv_data)
{
    if (get_variant_value() & (TEST_TAG | TEST_STREAM)) {
        for (int i = 0; i < repeat; ++i) {
            size_t recv_length;
            void *req;

            clear_recv_data();
            if (get_variant_value() & TEST_TAG) {
                req = ucp_tag_recv_nb(worker, &m_recv_data[0], length, DT_U64,
                                      TAG, (ucp_tag_t)-1, tag_recv_completion);
            } else if (get_variant_value() & TEST_STREAM) {
                req = ucp_stream_recv_nb(ep, &m_recv_data[0], length, DT_U64,
                                         stream_recv_completion, &recv_length,
                                         UCP_STREAM_RECV_FLAG_WAITALL);
            } else {
                req = NULL;
            }
            if (UCS_PTR_IS_PTR(req)) {
                request_wait(req);
            } else {
                ASSERT_UCS_OK(UCS_PTR_STATUS(req));
            }
            EXPECT_EQ(recv_data, m_recv_data[0])
                      << "repeat " << i << "/" << repeat;
            EXPECT_EQ(length,
                      (size_t)std::count(m_recv_data.begin(),
                                         m_recv_data.begin() + length,
                                         recv_data));
        }
    } else if (get_variant_value() & TEST_RMA) {
        for (size_t i = 0; i < length; ++i) {
            while (m_recv_data[i] != recv_data + repeat - 1) {
                progress();
            }
        }
    }
}

void test_ucp_wireup::send_completion(void *request, ucs_status_t status)
{
}

void test_ucp_wireup::close_completion(void *request, ucs_status_t status,
                                       void *user_data)
{
    ASSERT_UCS_OK(status);
    ASSERT_NE((test_ucp_wireup *)NULL, (test_ucp_wireup *)user_data);
}


void test_ucp_wireup::tag_recv_completion(void *request, ucs_status_t status,
                                          ucp_tag_recv_info_t *info)
{
}

void test_ucp_wireup::stream_recv_completion(void *request, ucs_status_t status,
                                             size_t length)
{
}

void test_ucp_wireup::send_recv(ucp_ep_h send_ep, ucp_worker_h recv_worker,
                                ucp_ep_h recv_ep, size_t length, int repeat)
{
    std::vector<void*> send_reqs;
    static uint64_t next_send_data = 0;
    uint64_t send_data = next_send_data++;

    send_nb(send_ep, length, repeat, send_reqs, send_data);
    recv_b (recv_worker, recv_ep, length, repeat, send_data);
    waitall(send_reqs);
    m_rkeys.clear();
}

void test_ucp_wireup::disconnect(ucp_ep_h ep) {
    void *req = ucp_disconnect_nb(ep);
    if (!UCS_PTR_IS_PTR(req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    }
    request_wait(req);
}

void test_ucp_wireup::disconnect(ucp_test::entity &e) {
    disconnect(e.revoke_ep());
}

void test_ucp_wireup::waitall(std::vector<void*> reqs)
{
    while (!reqs.empty()) {
        request_wait(reqs.back());
        reqs.pop_back();
    }
}

bool test_ucp_wireup::ep_iface_has_caps(const entity& e, const std::string& tl,
                                        uint64_t caps)
{
    ucp_worker_h worker = e.worker();
    ucp_context_h ctx   = worker->context;

    for (unsigned i = 0; i < worker->num_ifaces; ++i) {
        ucp_worker_iface_t *wiface = worker->ifaces[i];

        char* name = ctx->tl_rscs[wiface->rsc_index].tl_rsc.tl_name;
        if ((tl.empty() || !strcmp(name, tl.c_str())) &&
            ucs_test_all_flags(wiface->attr.cap.flags, caps)) {
            return true;
        }
    }

    return false;
}

class test_ucp_wireup_1sided : public test_ucp_wireup {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        test_ucp_wireup::get_test_variants(variants,
                                           UCP_FEATURE_RMA | UCP_FEATURE_TAG);
    }

    test_ucp_wireup_1sided() {
        for (ucp_lane_index_t i = 0; i < UCP_MAX_LANES; ++i) {
            m_lanes2remote[i] = i;
        }
    }

    ucp_lane_index_t m_lanes2remote[UCP_MAX_LANES];
};

UCS_TEST_P(test_ucp_wireup_1sided, address) {
    ucs_status_t status;
    size_t size;
    void *buffer;
    std::set<uint8_t> packed_dev_priorities, unpacked_dev_priorities;
    ucp_rsc_index_t tl;

    status = ucp_address_pack(sender().worker(), NULL,
                              std::numeric_limits<uint64_t>::max(),
                              UCP_ADDRESS_PACK_FLAGS_ALL, m_lanes2remote, &size,
                              &buffer);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(buffer != NULL);
    ASSERT_GT(size, 0ul);
    EXPECT_LE(size, 2048ul); /* Expect a reasonable address size */

    ucs_for_each_bit(tl, sender().worker()->context->tl_bitmap) {
        if (sender().worker()->context->tl_rscs[tl].flags & UCP_TL_RSC_FLAG_SOCKADDR) {
            continue;
        }
        packed_dev_priorities.insert(ucp_worker_iface_get_attr(sender().worker(), tl)->priority);
    }

    ucp_unpacked_address unpacked_address;

    status = ucp_address_unpack(sender().worker(), buffer,
                                UCP_ADDRESS_PACK_FLAGS_ALL, &unpacked_address);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sender().worker()->uuid, unpacked_address.uuid);
#if ENABLE_DEBUG_DATA
    EXPECT_EQ(std::string(ucp_worker_get_name(sender().worker())),
              std::string(unpacked_address.name));
#endif
    EXPECT_LE(unpacked_address.address_count,
              static_cast<unsigned>(sender().ucph()->num_tls));

    const ucp_address_entry_t *ae;
    ucp_unpacked_address_for_each(ae, &unpacked_address) {
        unpacked_dev_priorities.insert(ae->iface_attr.priority);
    }

    /* TODO test addresses */

    ucs_free(unpacked_address.address_list);
    ucs_free(buffer);
    /* Make sure that the packed device priorities are equal to the unpacked
     * device priorities */
    ASSERT_TRUE(packed_dev_priorities == unpacked_dev_priorities);
}

UCS_TEST_P(test_ucp_wireup_1sided, ep_address, "IB_NUM_PATHS?=2") {
    ucs_status_t status;
    size_t size;
    void *buffer;

    sender().connect(&receiver(), get_ep_params());

    status = ucp_address_pack(sender().worker(), sender().ep(),
                              std::numeric_limits<uint64_t>::max(),
                              UCP_ADDRESS_PACK_FLAGS_ALL, m_lanes2remote, &size,
                              &buffer);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(buffer != NULL);

    ucp_unpacked_address unpacked_address;

    status = ucp_address_unpack(sender().worker(), buffer,
                                UCP_ADDRESS_PACK_FLAGS_ALL, &unpacked_address);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sender().worker()->uuid, unpacked_address.uuid);
    EXPECT_LE(unpacked_address.address_count,
              static_cast<unsigned>(sender().ucph()->num_tls));

    ucs_free(unpacked_address.address_list);
    ucs_free(buffer);
}

UCS_TEST_P(test_ucp_wireup_1sided, empty_address) {
    ucs_status_t status;
    size_t size;
    void *buffer;

    status = ucp_address_pack(sender().worker(), NULL, 0,
                              UCP_ADDRESS_PACK_FLAGS_ALL, m_lanes2remote, &size,
                              &buffer);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(buffer != NULL);
    ASSERT_GT(size, 0ul);

    ucp_unpacked_address unpacked_address;

    status = ucp_address_unpack(sender().worker(), buffer,
                                UCP_ADDRESS_PACK_FLAGS_ALL, &unpacked_address);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sender().worker()->uuid, unpacked_address.uuid);
#if ENABLE_DEBUG_DATA
    EXPECT_EQ(std::string(ucp_worker_get_name(sender().worker())),
              std::string(unpacked_address.name));
#endif
    EXPECT_EQ(0u, unpacked_address.address_count);

    ucs_free(unpacked_address.address_list);
    ucs_free(buffer);
}

UCS_TEST_P(test_ucp_wireup_1sided, one_sided_wireup) {
    sender().connect(&receiver(), get_ep_params());
    send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
    flush_worker(sender());
}

UCS_TEST_P(test_ucp_wireup_1sided, one_sided_wireup_rndv, "RNDV_THRESH=1") {
    sender().connect(&receiver(), get_ep_params());
    send_recv(sender().ep(), receiver().worker(), receiver().ep(), BUFFER_LENGTH, 1);
    if (is_loopback() && (get_variant_value() & TEST_TAG)) {
        /* expect the endpoint to be connected to itself */
        ucp_ep_h ep         = sender().ep();
        ucp_worker_h worker = sender().worker();
        EXPECT_EQ(ep, ucp_worker_get_ep_by_id(worker, ucp_ep_remote_id(ep)));
    }
    flush_worker(sender());
}

UCS_TEST_P(test_ucp_wireup_1sided, multi_wireup) {
    skip_loopback();

    const size_t count = 10;
    while (entities().size() < count) {
        create_entity();
    }

    /* connect from sender() to all the rest */
    for (size_t i = 0; i < count; ++i) {
        sender().connect(&entities().at(i), get_ep_params(), i);
    }
}

UCS_TEST_P(test_ucp_wireup_1sided, stress_connect) {
    for (int i = 0; i < 30; ++i) {
        sender().connect(&receiver(), get_ep_params());
        send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1,
                  10000 / (ucs::test_time_multiplier() *
                           ucs::test_time_multiplier()));
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }

        disconnect(sender());
        if (!is_loopback()) {
            disconnect(receiver());
        }
    }
}

UCS_TEST_P(test_ucp_wireup_1sided, stress_connect2) {
    int max_count = (int)ucs_max(10,
                                 (1000.0 / (ucs::test_time_multiplier() *
                                            ucs::test_time_multiplier())));
    int count     = ucs_min(max_count, max_connections() / 2);

    for (int i = 0; i < count; ++i) {
        sender().connect(&receiver(), get_ep_params());
        send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }

        disconnect(sender());
        if (!is_loopback()) {
            disconnect(receiver());
        }
    }
}

UCS_TEST_P(test_ucp_wireup_1sided, disconnect_nonexistent) {
    skip_loopback();
    sender().connect(&receiver(), get_ep_params());
    disconnect(sender());
    receiver().destroy_worker();
    sender().destroy_worker();
}

UCS_TEST_P(test_ucp_wireup_1sided, disconnect_reconnect) {
    sender().connect(&receiver(), get_ep_params());
    send_b(sender().ep(), 1000, 1);
    disconnect(sender());
    recv_b(receiver().worker(), receiver().ep(), 1000, 1);

    sender().connect(&receiver(), get_ep_params());
    send_b(sender().ep(), 1000, 1);
    disconnect(sender());
    recv_b(receiver().worker(), receiver().ep(), 1000, 1);
}

UCS_TEST_P(test_ucp_wireup_1sided, send_disconnect_onesided) {
    sender().connect(&receiver(), get_ep_params());
    send_b(sender().ep(), 1000, 100);
    disconnect(sender());
    recv_b(receiver().worker(), receiver().ep(), 1000, 100);
}

UCS_TEST_P(test_ucp_wireup_1sided, send_disconnect_onesided_nozcopy, "ZCOPY_THRESH=-1") {
    sender().connect(&receiver(), get_ep_params());
    send_b(sender().ep(), 1000, 100);
    disconnect(sender());
    recv_b(receiver().worker(), receiver().ep(), 1000, 100);
}

UCS_TEST_P(test_ucp_wireup_1sided, send_disconnect_onesided_wait) {
    sender().connect(&receiver(), get_ep_params());
    send_recv(sender().ep(), receiver().worker(), receiver().ep(), 8, 1);
    send_b(sender().ep(), 1000, 200);
    disconnect(sender());
    recv_b(receiver().worker(), receiver().ep(), 1000, 200);
}

UCS_TEST_P(test_ucp_wireup_1sided, send_disconnect_reply1) {
    sender().connect(&receiver(), get_ep_params());
    if (!is_loopback()) {
        receiver().connect(&sender(), get_ep_params());
    }

    send_b(sender().ep(), 8, 1);
    if (!is_loopback()) {
        disconnect(sender());
    }

    recv_b(receiver().worker(), receiver().ep(), 8, 1);
    send_b(receiver().ep(), 8, 1);
    disconnect(receiver());
    recv_b(sender().worker(), sender().ep(), 8, 1);
}

UCS_TEST_SKIP_COND_P(test_ucp_wireup_1sided, send_disconnect_reply2,
                     /* skip the test for TCP, because it fails from time to
                      * time: the sender re-uses a socket fd from the already
                      * accepted connection from the receiver, but then the
                      * socket fd is closed, since the receiver closed the
                      * connection and the underlying TCP EP isn't able to
                      * receive the data on the failed socket.
                      * TODO: fix the bug on TCP level */
                     has_transport("tcp")) {
    sender().connect(&receiver(), get_ep_params());

    send_b(sender().ep(), 8, 1);
    if (!is_loopback()) {
        disconnect(sender());
    }
    recv_b(receiver().worker(), receiver().ep(),  8, 1);

    if (!is_loopback()) {
        receiver().connect(&sender(), get_ep_params());
    }

    send_b(receiver().ep(), 8, 1);
    disconnect(receiver());
    recv_b(sender().worker(), receiver().ep(), 8, 1);
}

UCS_TEST_P(test_ucp_wireup_1sided, disconnect_nb_onesided) {
    sender().connect(&receiver(), get_ep_params());

    std::vector<void*> sreqs;
    send_nb(sender().ep(), 1000, 1000, sreqs);

    void *req = sender().disconnect_nb();
    ucs_time_t deadline = ucs::get_deadline();
    while (!is_request_completed(req) && (ucs_get_time() < deadline)) {
        progress();
    }

    sender().close_ep_req_free(req);

    recv_b(receiver().worker(), receiver().ep(), 1000, 1000);
    waitall(sreqs);
}

UCS_TEST_P(test_ucp_wireup_1sided, multi_ep_1sided) {
    const unsigned count = 10;

    for (unsigned i = 0; i < count; ++i) {
        sender().connect(&receiver(), get_ep_params(), i);
    }

    for (unsigned i = 0; i < count; ++i) {
        send_recv(sender().ep(0, i), receiver().worker(), receiver().ep(), 8, 1);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_1sided)

class test_ucp_wireup_2sided : public test_ucp_wireup {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        test_ucp_wireup::get_test_variants(variants, UCP_FEATURE_RMA |
                                           UCP_FEATURE_TAG | UCP_FEATURE_STREAM);
    }

protected:
    void test_connect_loopback(bool delay_before_connect, bool enable_loopback);
};

UCS_TEST_P(test_ucp_wireup_2sided, two_sided_wireup) {
    sender().connect(&receiver(), get_ep_params());
    if (!is_loopback()) {
        receiver().connect(&sender(), get_ep_params());
    }

    send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
    flush_worker(sender());
    send_recv(receiver().ep(), sender().worker(), sender().ep(), 1, 1);
    flush_worker(receiver());
}

void test_ucp_wireup_2sided::test_connect_loopback(bool delay_before_connect,
                                                   bool enable_loopback) {
    ucp_ep_params_t params = test_ucp_wireup::get_ep_params();
    if (!enable_loopback) {
        params.field_mask |= UCP_EP_PARAM_FIELD_FLAGS;
        params.flags      |= UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
    }

    for (int i = 0; i < 5; ++i) {
        int base_index = i * 2;
        sender().connect(&sender(), params, base_index);
        ucp_ep_h ep1 = sender().ep(0, base_index);

        if (delay_before_connect) {
            /* let one side create ep */
            short_progress_loop(0);
        }

        sender().connect(&sender(), params, base_index + 1);
        ucp_ep_h ep2 = sender().ep(0, base_index + 1);

        EXPECT_NE(ep1, ep2);

        if (get_variant_value() & TEST_STREAM) {
            uint64_t data1 = (base_index * 10) + 1;
            uint64_t data2 = (base_index * 10) + 2;

            send_b(ep1, 1, 1, data1);
            send_b(ep2, 1, 1, data2);

            if (enable_loopback) {
                /* self-send - each ep receives what was sent on it */
                recv_b(sender().worker(), ep1, 1, 1, data1);
                recv_b(sender().worker(), ep2, 1, 1, data2);
            } else {
                /* cross-send - each ep receives what was sent on the other ep */
                recv_b(sender().worker(), ep1, 1, 1, data2);
                recv_b(sender().worker(), ep2, 1, 1, data1);
            }
        }
    }
    flush_worker(sender());
}

UCS_TEST_P(test_ucp_wireup_2sided, loopback) {
    test_connect_loopback(false, true);
}

UCS_TEST_P(test_ucp_wireup_2sided, loopback_with_delay) {
    test_connect_loopback(true, true);
}

UCS_TEST_P(test_ucp_wireup_2sided, no_loopback) {
    test_connect_loopback(false, false);
}

UCS_TEST_P(test_ucp_wireup_2sided, no_loopback_with_delay) {
    test_connect_loopback(true, false);
}

UCS_TEST_SKIP_COND_P(test_ucp_wireup_2sided, async_connect,
                     !(get_variant_ctx_params().features & UCP_FEATURE_TAG)) {
    sender().connect(&receiver(), get_ep_params());
    ucp_ep_h send_ep = sender().ep();
    std::vector<void *> reqs;

    reqs.push_back(ucp_tag_send_nb(send_ep, NULL, 0, DT_U64, 1, send_completion));
    EXPECT_FALSE(UCS_PTR_IS_ERR(reqs.back()));

    ucs_time_t deadline = ucs::get_deadline();
    /* waiting of async reply on wiriup without calling progress on receiver */
    while(!(send_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED) &&
          (ucs_get_time() < deadline)) {
        ucp_worker_progress(sender().worker());
        ucp_worker_progress(receiver().worker());
    }

    reqs.push_back(ucp_tag_recv_nb(receiver().worker(), NULL, 0, DT_U64, 1,
                                   (ucp_tag_t)-1, tag_recv_completion));
    EXPECT_FALSE(UCS_PTR_IS_ERR(reqs.back()));
    waitall(reqs);
}

UCS_TEST_P(test_ucp_wireup_2sided, connect_disconnect) {
    sender().connect(&receiver(), get_ep_params());
    if (!is_loopback()) {
        receiver().connect(&sender(), get_ep_params());
    }
    disconnect(sender());
    if (!is_loopback()) {
        disconnect(receiver());
    }
}

UCS_TEST_P(test_ucp_wireup_2sided, close_nbx_callback) {
    sender().connect(&receiver(), get_ep_params());
    if (!is_loopback()) {
        receiver().connect(&sender(), get_ep_params());
    }

    std::vector<void *> reqs;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK  |
                         UCP_OP_ATTR_FIELD_USER_DATA |
                         UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    param.cb.send      = close_completion;
    param.user_data    = this;

    reqs.push_back(ucp_ep_close_nbx(sender().revoke_ep(), &param));
    EXPECT_FALSE(UCS_PTR_IS_ERR(reqs.back()));

    if (!is_loopback()) {
        reqs.push_back(ucp_ep_close_nbx(receiver().revoke_ep(), &param));
        EXPECT_FALSE(UCS_PTR_IS_ERR(reqs.back()));
    }

    waitall(reqs);
}

UCS_TEST_P(test_ucp_wireup_2sided, multi_ep_2sided) {
    const unsigned count = 10;

    for (unsigned j = 0; j < 4; ++j) {

        unsigned offset = j * count;

        for (unsigned i = 0; i < count; ++i) {
            unsigned ep_idx = offset + i;
            sender().connect(&receiver(), get_ep_params(), ep_idx);
            if (!is_loopback()) {
                receiver().connect(&sender(), get_ep_params(), ep_idx);
            }
            UCS_TEST_MESSAGE << "iteration " << j << " pair " << i << ": " <<
                            sender().ep(0, ep_idx) << " <--> " << receiver().ep(0, ep_idx);
        }

        for (unsigned i = 0; i < count; ++i) {
            unsigned ep_idx = offset + i;
            send_recv(sender().ep(0, ep_idx), receiver().worker(),
                      receiver().ep(0, ep_idx), 8, 1);
            send_recv(receiver().ep(0, ep_idx), sender().worker(),
                      sender().ep(0, ep_idx), 8, 1);
        }

        short_progress_loop(0);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_2sided)

class test_ucp_wireup_errh_peer : public test_ucp_wireup_1sided
{
public:
    virtual ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params = test_ucp_wireup::get_ep_params();
        params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER;
        params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        params.err_handler.cb  = err_cb;
        params.err_handler.arg = NULL;
        return params;
    }

    virtual void init() {
        test_ucp_wireup::init();
        skip_loopback();
    }

    static void err_cb(void *, ucp_ep_h, ucs_status_t) {}
};

UCS_TEST_P(test_ucp_wireup_errh_peer, msg_after_ep_create) {
    receiver().connect(&sender(), get_ep_params());

    sender().connect(&receiver(), get_ep_params());
    send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
    flush_worker(sender());
}

UCS_TEST_P(test_ucp_wireup_errh_peer, msg_before_ep_create) {

    sender().connect(&receiver(), get_ep_params());
    send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
    flush_worker(sender());

    receiver().connect(&sender(), get_ep_params());

    send_recv(receiver().ep(), sender().worker(), receiver().ep(), 1, 1);
    flush_worker(receiver());
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_errh_peer)

class test_ucp_wireup_fallback : public test_ucp_wireup {
public:
    test_ucp_wireup_fallback() {
        m_num_lanes = 0;
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        test_ucp_wireup::get_test_variants(variants, UCP_FEATURE_RMA |
                                           UCP_FEATURE_TAG | UCP_FEATURE_STREAM);
    }

    void init() {
        /* do nothing */
    }

    void cleanup() {
        /* do nothing */
    }

    bool check_scalable_tls(const ucp_worker_h worker, size_t est_num_eps) {
        ucp_rsc_index_t rsc_index;

        ucs_for_each_bit(rsc_index, worker->context->tl_bitmap) {
            ucp_md_index_t md_index      = worker->context->tl_rscs[rsc_index].md_index;
            const uct_md_attr_t *md_attr = &worker->context->tl_mds[md_index].attr;

            if ((worker->context->tl_rscs[rsc_index].flags & UCP_TL_RSC_FLAG_AUX) ||
                (md_attr->cap.flags & UCT_MD_FLAG_SOCKADDR) ||
                (worker->context->tl_rscs[rsc_index].tl_rsc.dev_type == UCT_DEVICE_TYPE_ACC)) {
                // Skip TLs for wireup and CM and acceleration TLs
                continue;
            }

            if (ucp_worker_iface_get_attr(worker, rsc_index)->max_num_eps >= est_num_eps) {
                EXPECT_TRUE((worker->scalable_tl_bitmap & UCS_BIT(rsc_index)) != 0);
                return true;
            } else {
                EXPECT_TRUE((worker->scalable_tl_bitmap & UCS_BIT(rsc_index)) == 0);
            }
        }

        return false;
    }

    bool test_est_num_eps_fallback(size_t est_num_eps,
                                   unsigned long &min_max_num_eps) {
        size_t num_lanes = 0;
        bool res         = true;
        bool has_only_unscalable;

        min_max_num_eps = UCS_ULUNITS_INF;

        UCS_TEST_MESSAGE << "Testing " << est_num_eps << " number of EPs";
        modify_config("NUM_EPS", ucs::to_string(est_num_eps).c_str());
        test_ucp_wireup::init();

        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }
        send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
        flush_worker(sender());

        has_only_unscalable = !check_scalable_tls(sender().worker(),
                                                  est_num_eps);

        for (ucp_lane_index_t lane = 0;
             lane < ucp_ep_num_lanes(sender().ep()); lane++) {
            uct_ep_h uct_ep = sender().ep()->uct_eps[lane];
            if (uct_ep == NULL) {
                continue;
            }

            uct_iface_attr_t iface_attr;
            ucs_status_t status = uct_iface_query(uct_ep->iface, &iface_attr);
            ASSERT_UCS_OK(status);

            num_lanes++;

            if (!has_only_unscalable && (iface_attr.max_num_eps < est_num_eps)) {
                res = false;
                goto out;
            }

            if (iface_attr.max_num_eps < min_max_num_eps) {
                min_max_num_eps = iface_attr.max_num_eps;
            }
        }

out:
        test_ucp_wireup::cleanup();

        if (est_num_eps == 1) {
            m_num_lanes = num_lanes;
        } else if (has_only_unscalable) {
            /* If has only unscalable transports, check that the number of
             * lanes is the same as for the case when "est_num_eps == 1" */
            res = (num_lanes == m_num_lanes);
        }

        return res;
    }

private:

    /* The number of lanes activated for the case when "est_num_eps == 1" */
    size_t m_num_lanes;
};

UCS_TEST_P(test_ucp_wireup_fallback, est_num_eps_fallback) {
    unsigned long test_min_max_eps, min_max_eps;

    test_est_num_eps_fallback(1, test_min_max_eps);

    size_t prev_min_max_eps = 0;
    while ((test_min_max_eps != UCS_ULUNITS_INF) &&
           /* number of EPs was changed between iterations */
           (test_min_max_eps != prev_min_max_eps)) {
        if (test_min_max_eps > 1) {
            EXPECT_TRUE(test_est_num_eps_fallback(test_min_max_eps - 1,
                                                  min_max_eps));
        }

        EXPECT_TRUE(test_est_num_eps_fallback(test_min_max_eps,
                                              min_max_eps));

        EXPECT_TRUE(test_est_num_eps_fallback(test_min_max_eps + 1,
                                              min_max_eps));
        prev_min_max_eps = test_min_max_eps;
        test_min_max_eps = min_max_eps;
    }
}

/* Test fallback from RC to UD, since RC isn't scalable enough
 * as its iface max_num_eps attribute = 256 by default */
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_fallback,
                              rc_ud, "rc_x,rc_v,ud_x,ud_v")
/* Test fallback selection of UD only TLs, since TCP shouldn't
 * be used for any lanes */
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_fallback,
                              ud_tcp, "ud_x,ud_v,tcp")
/* Test two scalable enough transports */
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_fallback,
                              dc_ud, "dc_x,ud_x,ud_v")
/* Test unsacalable transports only */
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_fallback,
                              rc, "rc_x,rc_v")
/* Test all available IB transports */
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_fallback,
                              ib, "ib")
/* Test on TCP only */
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_fallback,
                              tcp, "tcp")

class test_ucp_wireup_unified : public test_ucp_wireup {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG,
                               TEST_TAG | UNIFIED_MODE, "uni");
    }

    bool context_has_tls(ucp_context_h ctx, const std::string& tl,
                         ucp_rsc_index_t md_idx)
    {
        for (ucp_rsc_index_t idx = 0; idx < ctx->num_tls; ++idx) {
            if (ctx->tl_rscs[idx].md_index != md_idx) {
                continue;
            }

            if (!strcmp(ctx->tl_rscs[idx].tl_rsc.tl_name, tl.c_str())) {
                return true;
            }
        }

        return false;
    }

    bool worker_has_tls(ucp_worker_h worker, const std::string& tl,
                        ucp_rsc_index_t md_idx)
    {
        ucp_context_h ctx = worker->context;

        for (unsigned i = 0; i < worker->num_ifaces; ++i) {
            ucp_worker_iface_t *wiface = worker->ifaces[i];
            ucp_rsc_index_t md_idx_it  = ctx->tl_rscs[wiface->rsc_index].md_index;

            if (md_idx_it != md_idx) {
                continue;
            }

            char* name = ctx->tl_rscs[wiface->rsc_index].tl_rsc.tl_name;
            if (!strcmp(name, tl.c_str())) {
                return true;
            }
        }

        return false;
    }

    void check_unified_ifaces(entity *e,
                              const std::string& better_tl,
                              const std::string& tl)
    {
        ucp_context_h ctx   = e->ucph();
        ucp_worker_h worker = e->worker();

        for (ucp_rsc_index_t i = 0; i < ctx->num_mds; ++i) {
            if (!(context_has_tls(ctx, better_tl, i) &&
                  context_has_tls(ctx, tl, i))) {
               continue;
            }

            ASSERT_TRUE(ctx->num_tls > worker->num_ifaces);
            EXPECT_TRUE(worker_has_tls(worker, better_tl, i)) <<
                " transport " << better_tl << " should not be closed";
            EXPECT_FALSE(worker_has_tls(worker, tl, i)) <<
                " transport " << better_tl << " should be closed";
        }
    }
};


UCS_TEST_P(test_ucp_wireup_unified, select_best_ifaces)
{
    // Accelerated transports have better performance charasteristics than their
    // verbs counterparts. Check that corresponding verbs transports are not used
    // by workers in unified mode.
    check_unified_ifaces(&sender(), "rc_mlx5", "rc_verbs");
    check_unified_ifaces(&sender(), "ud_mlx5", "ud_verbs");

    // RC and DC has similar capabilities, but RC has better latency while
    // estimated number of endpoints is relatively small.
    // sender() is created with 1 ep, so RC should be selected over DC.
    check_unified_ifaces(&sender(), "rc_mlx5", "dc_mlx5");

    // Set some big enough number of endpoints for DC to be more performance
    // efficient than RC. Now check that DC is selected over RC.
    // TODO: enable test when keepalive feature is enabled for DC transport
    //modify_config("NUM_EPS", "1000");
    //entity *e = create_entity();
    //check_unified_ifaces(e, "dc_mlx5", "rc_mlx5");
    EXPECT_FALSE(ep_iface_has_caps(sender(), "dc_mlx5",
                                   UCT_IFACE_FLAG_EP_CHECK));
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_unified, rc, "rc")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_unified, ud, "ud")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_unified, rc_dc, "rc,dc")

class test_ucp_wireup_fallback_amo : public test_ucp_wireup {
protected:
    void init() {
        size_t device_atomics_cnt = 0;

        test_ucp_wireup::init();

        for (ucp_rsc_index_t idx = 0; idx < sender().ucph()->num_tls; ++idx) {
            uct_iface_attr_t *attr = ucp_worker_iface_get_attr(sender().worker(),
                                                               idx);
            if (attr->cap.flags & UCT_IFACE_FLAG_ATOMIC_DEVICE) {
                device_atomics_cnt++;
            }
        }
        bool device_atomics_supported = sender().worker()->atomic_tls != 0;

        test_ucp_wireup::cleanup();

        if (!device_atomics_supported || !device_atomics_cnt) {
            UCS_TEST_SKIP_R("there are no TLs that support device atomics");
        }
    }

    void cleanup() {
        /* do nothing */
    }

    bool use_device_amo(ucp_ep_h ep) {
        ucp_ep_config_t *ep_config = ucp_ep_config(ep);

        for (ucp_lane_index_t lane = 0; lane < UCP_MAX_LANES; ++lane) {
            if (ep_config->key.amo_lanes[lane] != UCP_NULL_LANE) {
                return (ucp_ep_get_iface_attr(ep, lane)->cap.flags &
                        UCT_IFACE_FLAG_ATOMIC_DEVICE);
            }
        }

        return false;
    }

    size_t get_min_max_num_eps(ucp_ep_h ep) {
        unsigned long min_max_num_eps = UCS_ULUNITS_INF;

        for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep); lane++) {
            uct_iface_attr_t *iface_attr = ucp_ep_get_iface_attr(ep, lane);

            if (iface_attr->max_num_eps < min_max_num_eps) {
                min_max_num_eps = iface_attr->max_num_eps;
            }
        }

        return min_max_num_eps;
    }

    size_t test_wireup_fallback_amo(const std::vector<std::string> &tls,
                                    size_t est_num_eps, bool should_use_device_amo) {
        unsigned long min_max_num_eps = UCS_ULUNITS_INF;

        UCS_TEST_MESSAGE << "Testing " << est_num_eps << " number of EPs";
        modify_config("NUM_EPS", ucs::to_string(est_num_eps).c_str());

        // Create new entity and add to to the end of vector
        // (thus it will be receiver without any connections)
        create_entity(false);

        ucp_test_param params = GetParam();
        for (std::vector<std::string>::const_iterator i = tls.begin();
             i != tls.end(); ++i) {
            params.transports.clear();
            params.transports.push_back(*i);
            create_entity(true, params);
            sender().connect(&receiver(), get_ep_params());

            EXPECT_EQ(should_use_device_amo, use_device_amo(sender().ep()));

            size_t max_num_eps = get_min_max_num_eps(sender().ep());
            if (max_num_eps < min_max_num_eps) {
                min_max_num_eps = max_num_eps;
            }
        }

        test_ucp_wireup::cleanup();

        return min_max_num_eps;
    }

public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        test_ucp_wireup::get_test_variants(variants, UCP_FEATURE_AMO32 |
                                           UCP_FEATURE_AMO64);
    }
};

class test_ucp_wireup_amo : public test_ucp_wireup {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        UCS_STATIC_ASSERT((sizeof(elem_type) == sizeof(uint32_t)) ||
                          (sizeof(elem_type) == sizeof(uint64_t)));
        uint64_t amo_features = (sizeof(elem_type) == sizeof(uint32_t)) ?
                                UCP_FEATURE_AMO32 : UCP_FEATURE_AMO64;
        test_ucp_wireup::get_test_variants(variants, amo_features);
    }

protected:
    ucp_rkey_h get_rkey(const entity &e) {
        if (&sender() == &e) {
            return test_ucp_wireup::get_rkey(e.ep(), m_memh_receiver);
        } else if (&receiver() == &e) {
            return test_ucp_wireup::get_rkey(e.ep(), m_memh_sender);
        }

        return NULL;
    }

    void add_rkey(ucp_rkey_h rkey) {
        ASSERT_NE((ucp_rkey_h)NULL, rkey);
        m_rkeys.push_back(ucs::handle<ucp_rkey_h>(rkey, ucp_rkey_destroy));
    }

    void fill_send_data() {
        m_send_data[0] = ucs_generate_uuid(0);
    }

    static void flush_cb(void *req, ucs_status_t status, void *user_data) {
        test_ucp_wireup_amo *test = (test_ucp_wireup_amo*)user_data;

        ASSERT_UCS_OK(status);
        test->rkeys_cleanup();
        test->memhs_cleanup();
    }
};

UCS_TEST_P(test_ucp_wireup_amo, relese_key_after_flush) {
    fill_send_data();
    clear_recv_data();

    sender().connect(&receiver(), get_ep_params());

    ucp_rkey_h rkey = get_rkey(sender());
    add_rkey(rkey);

    ucs_status_t status = ucp_atomic_post(sender().ep(), UCP_ATOMIC_POST_OP_ADD,
                                          m_send_data[0], sizeof(elem_type),
                                          (uint64_t)&m_recv_data[0], rkey);
    ASSERT_UCS_OK(status);

    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send      = flush_cb;
    param.user_data    = this;
    void *req = ucp_ep_flush_nbx(sender().ep(), &param);
    if (UCS_PTR_IS_PTR(req)) {
        request_wait(req);
    } else {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_amo)

UCS_TEST_P(test_ucp_wireup_fallback_amo, different_amo_types) {
    std::vector<std::string> tls;

    /* the 1st peer support RC only (device atomics) */
    tls.push_back("rc");
    /* the 2nd peer support RC and SHM (device and CPU atomics) */
    tls.push_back("rc,shm");

    size_t min_max_num_eps = test_wireup_fallback_amo(tls, 1, 1);
    test_wireup_fallback_amo(tls, min_max_num_eps + 1, 0);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_fallback_amo,
                              shm_rc, "shm,rc_x,rc_v")

/* NOTE: this fixture is NOT inherited from test_ucp_wireup, because we want to
 * create our own entities.
 */
class test_ucp_wireup_asymmetric : public ucp_test {
protected:
    virtual void init() {
        static const char *ibdev_sysfs_dir = "/sys/class/infiniband";

        DIR *dir = opendir(ibdev_sysfs_dir);
        if (dir == NULL) {
            UCS_TEST_SKIP_R(std::string(ibdev_sysfs_dir) + " not found");
        }

        for (;;) {
            struct dirent *entry = readdir(dir);
            if (entry == NULL) {
                break;
            }

            if (entry->d_name[0] == '.') {
                continue;
            }

            m_ib_devices.push_back(entry->d_name);
        }

        closedir(dir);
    }

    void tag_sendrecv(size_t size) {
        std::string send_data(size, 's');
        std::string recv_data(size, 'x');

        ucs_status_ptr_t sreq = ucp_tag_send_nb(
                        sender().ep(0), &send_data[0], size,
                        ucp_dt_make_contig(1), 1,
                        (ucp_send_callback_t)ucs_empty_function);
        ucs_status_ptr_t rreq = ucp_tag_recv_nb(
                        receiver().worker(), &recv_data[0], size,
                        ucp_dt_make_contig(1), 1, 1,
                        (ucp_tag_recv_callback_t)ucs_empty_function);
        request_wait(sreq);
        request_wait(rreq);

        EXPECT_EQ(send_data, recv_data);
    }

    /* Generate a pci_bw configuration string for IB devices, which assigns
     * the speed ai+b for device i.
     */
    std::string pci_bw_config(int a, int b) {
        std::string config_str;
        for (size_t i = 0; i < m_ib_devices.size(); ++i) {
            config_str += m_ib_devices[i] + ":" +
                            ucs::to_string((a * i) + b) + "Gbps";
            if (i != (m_ib_devices.size() - 1)) {
                config_str += ",";
            }
        }
        return config_str;
    }

    std::vector<std::string> m_ib_devices;

public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG);
    }
};

/*
 * Force asymmetric configuration by different PCI_BW settings
 */
UCS_TEST_SKIP_COND_P(test_ucp_wireup_asymmetric, connect, is_self()) {

    /* Enable cross-dev connection */
    /* coverity[tainted_string_argument] */
    ucs::scoped_setenv path_mtu_env("UCX_RC_PATH_MTU", "1024");

    {
        std::string config_str = pci_bw_config(20, 20);
        UCS_TEST_MESSAGE << "creating sender: " << config_str;
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv pci_bw_env("UCX_IB_PCI_BW", config_str.c_str());
        create_entity();
    }

    {
        std::string config_str = pci_bw_config(-20, m_ib_devices.size() * 20);
        UCS_TEST_MESSAGE << "creating receiver: " << config_str;
        /* coverity[tainted_string_argument] */
        ucs::scoped_setenv pci_bw_env("UCX_IB_PCI_BW", config_str.c_str());
        create_entity();
    }

    sender().connect(&receiver(), get_ep_params());
    receiver().connect(&sender(), get_ep_params());

    ucp_ep_print_info(sender().ep(), stdout);
    ucp_ep_print_info(receiver().ep(), stdout);

    tag_sendrecv(1);
    tag_sendrecv(100000);
    tag_sendrecv(1000000);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_asymmetric, rcv, "rc_v")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_asymmetric, rcx, "rc_x")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_wireup_asymmetric, ib, "ib")

class test_ucp_wireup_keepalive : public test_ucp_wireup {
public:
    test_ucp_wireup_keepalive() {
        m_env.push_back(new ucs::scoped_setenv("UCX_TCP_KEEPIDLE", "inf"));
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        test_ucp_wireup::get_test_variants(variants,
                                           UCP_FEATURE_RMA | UCP_FEATURE_TAG);
    }

    ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params;
        memset(&params, 0, sizeof(params));
        params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER;
        params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        params.err_handler.cb  = reinterpret_cast<ucp_err_handler_cb_t>
                                 (ucs_empty_function);
        params.err_handler.arg = reinterpret_cast<void*>(this);
        return params;
    }

    void init() {
        test_ucp_wireup::init();

        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }

protected:
    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};

/* test if EP has non-empty keepalive lanes mask */
UCS_TEST_P(test_ucp_wireup_keepalive, attr) {
    if (!sender().has_lane_with_caps(UCT_IFACE_FLAG_EP_CHECK)) {
        UCS_TEST_SKIP_R("Unsupported");
    }

    ucp_ep_config_t *ep_config = ucp_ep_config(sender().ep());
    EXPECT_NE(0, ep_config->key.ep_check_map);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_keepalive)
