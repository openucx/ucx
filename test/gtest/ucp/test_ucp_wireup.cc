/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#define __STDC_LIMIT_MACROS

#include "ucp_test.h"

#include <algorithm>
#include <set>

extern "C" {
#include <ucp/wireup/address.h>
#include <ucp/proto/proto.h>
}

class test_ucp_wireup : public ucp_test {
public:
    static std::vector<ucp_test_param>
    enum_test_params_features(const ucp_params_t& ctx_params,
                              const std::string& name,
                              const std::string& test_case_name,
                              const std::string& tls,
                              uint64_t features);

protected:
    enum {
        TEST_RMA,
        TEST_TAG,
        TEST_STREAM
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

private:
    vec_type                               m_send_data;
    vec_type                               m_recv_data;
    ucs::handle<ucp_mem_h, ucp_context_h>  m_memh_sender;
    ucs::handle<ucp_mem_h, ucp_context_h>  m_memh_receiver;
    std::vector< ucs::handle<ucp_rkey_h> > m_rkeys;

    void clear_recv_data();

    ucp_rkey_h get_rkey(ucp_ep_h ep, ucp_mem_h memh);

    static void send_completion(void *request, ucs_status_t status);

    static void tag_recv_completion(void *request, ucs_status_t status,
                                ucp_tag_recv_info_t *info);

    static void stream_recv_completion(void *request, ucs_status_t status,
                                       size_t length);

    static void unmap_memh(ucp_mem_h memh, ucp_context_h context);
};

std::vector<ucp_test_param>
test_ucp_wireup::enum_test_params_features(const ucp_params_t& ctx_params,
                                           const std::string& name,
                                           const std::string& test_case_name,
                                           const std::string& tls,
                                           uint64_t features)
{
    std::vector<ucp_test_param> result;
    ucp_params_t tmp_ctx_params = ctx_params;

    if (features & UCP_FEATURE_RMA) {
        tmp_ctx_params.features = UCP_FEATURE_RMA;
        generate_test_params_variant(tmp_ctx_params, name, test_case_name + "/rma",
                                     tls, TEST_RMA, result);
    }

    if (features & UCP_FEATURE_TAG) {
        tmp_ctx_params.features = UCP_FEATURE_TAG;
        generate_test_params_variant(tmp_ctx_params, name, test_case_name + "/tag",
                                     tls, TEST_TAG, result);
    }

    if (features & UCP_FEATURE_STREAM) {
        tmp_ctx_params.features = UCP_FEATURE_STREAM;
        generate_test_params_variant(tmp_ctx_params, name, test_case_name + "/stream",
                                     tls, TEST_STREAM, result);
    }

    return result;
}

void test_ucp_wireup::unmap_memh(ucp_mem_h memh, ucp_context_h context)
{
    ucs_status_t status = ucp_mem_unmap(context, memh);
    if (status != UCS_OK) {
        ucs_warn("failed to unmap memory: %s", ucs_status_string(status));
    }
}

void test_ucp_wireup::init() {
    ucp_test::init();

    m_send_data.resize(BUFFER_LENGTH, 0);
    m_recv_data.resize(BUFFER_LENGTH, 0);

    if (GetParam().variant == TEST_RMA) {
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

void test_ucp_wireup::cleanup() {
    m_rkeys.clear();
    m_memh_sender.reset();
    m_memh_receiver.reset();
    ucp_test::cleanup();
}

void test_ucp_wireup::clear_recv_data() {
    std::fill(m_recv_data.begin(), m_recv_data.end(), 0);
}

void test_ucp_wireup::send_nb(ucp_ep_h ep, size_t length, int repeat,
                              std::vector<void*>& reqs, uint64_t send_data)
{
    if (GetParam().variant == TEST_TAG) {
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
    } else if (GetParam().variant == TEST_STREAM) {
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
    } else if (GetParam().variant == TEST_RMA) {
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
    if ((GetParam().variant == TEST_TAG) || (GetParam().variant == TEST_STREAM))
    {
        for (int i = 0; i < repeat; ++i) {
            size_t recv_length;
            void *req;

            clear_recv_data();
            if (GetParam().variant == TEST_TAG) {
                req = ucp_tag_recv_nb(worker, &m_recv_data[0], length, DT_U64,
                                      TAG, (ucp_tag_t)-1, tag_recv_completion);
            } else if (GetParam().variant == TEST_STREAM) {
                req = ucp_stream_recv_nb(ep, &m_recv_data[0], length, DT_U64,
                                         stream_recv_completion, &recv_length,
                                         UCP_STREAM_RECV_FLAG_WAITALL);
            } else {
                req = NULL;
            }
            if (UCS_PTR_IS_PTR(req)) {
                wait(req);
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
    } else if (GetParam().variant == TEST_RMA) {
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
    wait(req);
}

void test_ucp_wireup::disconnect(ucp_test::entity &e) {
    disconnect(e.revoke_ep());
}

void test_ucp_wireup::waitall(std::vector<void*> reqs)
{
    while (!reqs.empty()) {
        wait(reqs.back());
        reqs.pop_back();
    }
}

class test_ucp_wireup_1sided : public test_ucp_wireup {
public:
    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params, const std::string& name,
                     const std::string& test_case_name, const std::string& tls)
    {
        return enum_test_params_features(ctx_params, name, test_case_name, tls,
                                         UCP_FEATURE_RMA | UCP_FEATURE_TAG);
    }
};

UCS_TEST_P(test_ucp_wireup_1sided, address) {
    ucs_status_t status;
    size_t size;
    void *buffer;
    unsigned order[UCP_MAX_RESOURCES];
    const ucp_address_entry_t *ae;
    std::set<uint8_t> packed_dev_priorities, unpacked_dev_priorities;
    int tl;

    status = ucp_address_pack(sender().worker(), NULL, -1, order, &size, &buffer);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(buffer != NULL);
    ASSERT_GT(size, 0ul);
    EXPECT_LE(size, 2048ul); /* Expect a reasonable address size */
    for (tl = 0; tl < sender().worker()->context->num_tls; tl++) {
        if (sender().worker()->context->tl_rscs[tl].flags & UCP_TL_RSC_FLAG_SOCKADDR) {
            continue;
        }
        packed_dev_priorities.insert(sender().worker()->ifaces[tl].attr.priority);
    }

    char name[UCP_WORKER_NAME_MAX];
    uint64_t uuid;
    unsigned address_count;
    ucp_address_entry_t *address_list;

    ucp_address_unpack(buffer, &uuid, name, sizeof(name), &address_count,
                       &address_list);
    EXPECT_EQ(sender().worker()->uuid, uuid);
    EXPECT_EQ(std::string(ucp_worker_get_name(sender().worker())), std::string(name));
    EXPECT_LE(address_count, static_cast<unsigned>(sender().ucph()->num_tls));
    for (ae = address_list; ae < address_list + address_count; ++ae) {
        unpacked_dev_priorities.insert(ae->iface_attr.priority);
    }

    /* TODO test addresses */

    ucs_free(address_list);
    ucs_free(buffer);
    /* Make sure that the packed device priorities are equal to the unpacked
     * device priorities */
    ASSERT_TRUE(packed_dev_priorities == unpacked_dev_priorities);
}

UCS_TEST_P(test_ucp_wireup_1sided, empty_address) {
    ucs_status_t status;
    size_t size;
    void *buffer;
    unsigned order[UCP_MAX_RESOURCES];

    status = ucp_address_pack(sender().worker(), NULL, 0, order, &size, &buffer);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(buffer != NULL);
    ASSERT_GT(size, 0ul);

    char name[UCP_WORKER_NAME_MAX];
    uint64_t uuid;
    unsigned address_count;
    ucp_address_entry_t *address_list;

    ucp_address_unpack(buffer, &uuid, name, sizeof(name), &address_count,
                       &address_list);
    EXPECT_EQ(sender().worker()->uuid, uuid);
    EXPECT_EQ(std::string(ucp_worker_get_name(sender().worker())), std::string(name));
    EXPECT_LE(address_count, sender().ucph()->num_tls);
    EXPECT_EQ(0u, address_count);

    ucs_free(address_list);
    ucs_free(buffer);
}

UCS_TEST_P(test_ucp_wireup_1sided, one_sided_wireup) {
    sender().connect(&receiver(), get_ep_params());
    send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
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

UCS_TEST_P(test_ucp_wireup_1sided, reply_ep_send_before) {
    skip_loopback();

    sender().connect(&receiver(), get_ep_params());

    if (GetParam().variant == TEST_TAG) {
        /* Send a reply */
        ucp_ep_connect_remote(sender().ep());
        ucp_ep_h ep = ucp_worker_get_reply_ep(receiver().worker(),
                                              sender().worker()->uuid);
        send_recv(ep, sender().worker(), sender().ep(), 1, 1);
        flush_worker(sender());

        disconnect(ep);
    }
}

UCS_TEST_P(test_ucp_wireup_1sided, reply_ep_send_after) {
    skip_loopback();

    sender().connect(&receiver(), get_ep_params());

    if (GetParam().variant == TEST_TAG) {
        ucp_ep_connect_remote(sender().ep());

        /* Make sure the wireup message arrives before sending a reply */
        send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
        flush_worker(sender());

        /* Send a reply */
        ucp_ep_h ep = ucp_worker_get_reply_ep(receiver().worker(),
                                              sender().worker()->uuid);
        send_recv(ep, sender().worker(), sender().ep(), 1, 1);

        flush_worker(sender());

        disconnect(ep);
    }
}

UCS_TEST_P(test_ucp_wireup_1sided, stress_connect) {
    for (int i = 0; i < 30; ++i) {
        sender().connect(&receiver(), get_ep_params());
        send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1,
                  10000 / ucs::test_time_multiplier());
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
    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        sender().connect(&receiver(), get_ep_params());
        send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
        if (&sender() != &receiver()) {
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

UCS_TEST_P(test_ucp_wireup_1sided, send_disconnect_reply2) {
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

    void *dreq = sender().disconnect_nb();
    if (!UCS_PTR_IS_PTR(dreq)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(dreq));
    }

    wait(dreq);
    recv_b(receiver().worker(), receiver().ep(), 1000, 1000);

    waitall(sreqs);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_1sided)

class test_ucp_wireup_2sided : public test_ucp_wireup {
public:
    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params, const std::string& name,
                     const std::string& test_case_name, const std::string& tls)
    {
        return enum_test_params_features(ctx_params, name, test_case_name, tls,
                                         UCP_FEATURE_RMA | UCP_FEATURE_TAG |
                                         UCP_FEATURE_STREAM);
    }
};

UCS_TEST_P(test_ucp_wireup_2sided, two_sided_wireup) {
    sender().connect(&receiver(), get_ep_params());
    if (&sender() != &receiver()) {
        receiver().connect(&sender(), get_ep_params());
    }

    send_recv(sender().ep(), receiver().worker(), receiver().ep(), 1, 1);
    flush_worker(sender());
    send_recv(receiver().ep(), sender().worker(), sender().ep(), 1, 1);
    flush_worker(receiver());
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
