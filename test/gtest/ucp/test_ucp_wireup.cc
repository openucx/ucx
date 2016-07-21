/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#define __STDC_LIMIT_MACROS

#include "ucp_test.h"

#include <algorithm>

extern "C" {
#include <ucp/core/ucp_worker.h>
#include <ucp/wireup/address.h>
#include <ucp/proto/proto.h>
}


class test_ucp_wireup_tag : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_TAG;
        return params;
    }

protected:
    void tag_send(ucp_ep_h from, ucp_worker_h to, int count = 1);

    static void send_completion(void *request, ucs_status_t status);

    static void recv_completion(void *request, ucs_status_t status,
                                ucp_tag_recv_info_t *info);
    void wait(void *req);
};

void test_ucp_wireup_tag::tag_send(ucp_ep_h from, ucp_worker_h to, int count)
{
    const ucp_datatype_t DATATYPE = ucp_dt_make_contig(1);
    const uint64_t TAG = 0xdeadbeef;
    uint64_t send_data = 0x12121212;
    std::vector<void*> reqs;
    void *req;

    for (int i = 0; i < count; ++i) {
        req = ucp_tag_send_nb(from, &send_data, sizeof(send_data), DATATYPE,
                              TAG, send_completion);
        if (UCS_PTR_IS_PTR(req)) {
            reqs.push_back(req);
        } else {
            ASSERT_UCS_OK(UCS_PTR_STATUS(req));
        }
    }

    for (int i = 0; i < count; ++i) {
        uint64_t recv_data = 0;
        req = ucp_tag_recv_nb(to, &recv_data, sizeof(recv_data),
                              DATATYPE, TAG, (ucp_tag_t)-1, recv_completion);
        wait(req);

        EXPECT_EQ(send_data, recv_data);
    }

    while (!reqs.empty()) {
        req = reqs.back();
        wait(req);
        reqs.pop_back();
    }
}

void test_ucp_wireup_tag::send_completion(void *request, ucs_status_t status)
{
}

void test_ucp_wireup_tag::recv_completion(void *request, ucs_status_t status,
                                      ucp_tag_recv_info_t *info)
{
}

void test_ucp_wireup_tag::wait(void *req)
{
    do {
        progress();
    } while (!ucp_request_is_completed(req));
    ucp_request_release(req);
}

UCS_TEST_P(test_ucp_wireup_tag, address) {
    ucs_status_t status;
    size_t size;
    void *buffer;
    unsigned order[UCP_MAX_RESOURCES];

    status = ucp_address_pack(sender().worker(), NULL, -1, order, &size, &buffer);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(buffer != NULL);
    ASSERT_GT(size, 0ul);
    EXPECT_LE(size, 512ul); /* Expect a reasonable address size */

    char name[UCP_WORKER_NAME_MAX];
    uint64_t uuid;
    unsigned address_count;
    ucp_address_entry_t *address_list;

    ucp_address_unpack(buffer, &uuid, name, sizeof(name), &address_count,
                       &address_list);
    EXPECT_EQ(sender().worker()->uuid, uuid);
    EXPECT_EQ(std::string(ucp_worker_get_name(sender().worker())), std::string(name));
    EXPECT_LE(address_count, static_cast<unsigned>(sender().ucph()->num_tls));

    /* TODO test addresses */

    ucs_free(address_list);
    ucs_free(buffer);
}

UCS_TEST_P(test_ucp_wireup_tag, empty_address) {
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


UCS_TEST_P(test_ucp_wireup_tag, one_sided_wireup) {
    sender().connect(&receiver());
    tag_send(sender().ep(), receiver().worker());
    sender().flush_worker();
}

UCS_TEST_P(test_ucp_wireup_tag, two_sided_wireup) {
    if (&sender() == &receiver()) {
        UCS_TEST_SKIP_R("loop-back unsupported");
    }

    sender().connect(&receiver());
    receiver().connect(&sender());

    tag_send(sender().ep(), receiver().worker());
    sender().flush_worker();
    tag_send(receiver().ep(), sender().worker());
    receiver().flush_worker();
}

UCS_TEST_P(test_ucp_wireup_tag, reply_ep_send_before) {
    if (&sender() == &receiver()) {
        UCS_TEST_SKIP_R("loop-back unsupported");
    }

    sender().connect(&receiver());

    ucp_ep_connect_remote(sender().ep());

    /* Send a reply */
    ucp_ep_h ep = ucp_worker_get_reply_ep(receiver().worker(), sender().worker()->uuid);
    tag_send(ep, sender().worker());
    sender().flush_worker();

    ucp_ep_destroy(ep);
}

UCS_TEST_P(test_ucp_wireup_tag, reply_ep_send_after) {
    if (&sender() == &receiver()) {
        UCS_TEST_SKIP_R("loop-back unsupported");
    }

    sender().connect(&receiver());

    ucp_ep_connect_remote(sender().ep());

    /* Make sure the wireup message arrives before sending a reply */
    tag_send(sender().ep(), receiver().worker());
    sender().flush_worker();

    /* Send a reply */
    ucp_ep_h ep = ucp_worker_get_reply_ep(receiver().worker(), sender().worker()->uuid);
    tag_send(ep, sender().worker());
    sender().flush_worker();

    ucp_ep_destroy(ep);
}

UCS_TEST_P(test_ucp_wireup_tag, stress_connect) {
    if (&sender() == &receiver()) {
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    for (int i = 0; i < 30; ++i) {
        sender().connect(&receiver());
        tag_send(sender().ep(), receiver().worker(), 10000 / ucs::test_time_multiplier());
        receiver().connect(&sender());
        sender().disconnect();
        receiver().disconnect();
    }
}

UCS_TEST_P(test_ucp_wireup_tag, stress_connect2) {
    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        sender().connect(&receiver());
        tag_send(sender().ep(), receiver().worker(), 1);
        if (&sender() != &receiver()) {
            receiver().connect(&sender());
        }
        sender().disconnect();
        receiver().disconnect();
    }
}

UCS_TEST_P(test_ucp_wireup_tag, connect_disconnect) {
    sender().connect(&receiver());
    if (&sender() != &receiver()) {
        receiver().connect(&sender());
    }
    sender().disconnect();
    receiver().disconnect();
}

UCS_TEST_P(test_ucp_wireup_tag, disconnect_nonexistent) {
    sender().connect(&receiver());
    receiver().destroy_worker();
    sender().disconnect();
}

class test_ucp_wireup_rma : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }

protected:
    void rma_send(ucp_ep_h from, ucp_worker_h to);
};

void test_ucp_wireup_rma::rma_send(ucp_ep_h from, ucp_worker_h to)
{
    uint64_t send_data = 0x12121212;
    uint64_t recv_data = 0;
    ucs_status_t status;
    void *rkey_buffer;
    size_t rkey_size;
    ucp_rkey_h rkey;
    ucp_mem_h memh;
    void *ptr;

    ptr = &recv_data;
    status = ucp_mem_map(to->context, &ptr, sizeof(recv_data), 0, &memh);
    ASSERT_UCS_OK(status);

    status = ucp_rkey_pack(to->context, memh, &rkey_buffer, &rkey_size);
    ASSERT_UCS_OK(status);

    status = ucp_ep_rkey_unpack(from, rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);

    status = ucp_put(from, &send_data, sizeof(send_data), (uintptr_t)&recv_data, rkey);
    ASSERT_UCS_OK(status);

    status = ucp_ep_flush(from);
    ASSERT_UCS_OK(status);

    while (recv_data != send_data);

    ucp_rkey_destroy(rkey);

    ucp_mem_unmap(to->context, memh);
}

UCS_TEST_P(test_ucp_wireup_rma, one_sided_wireup) {
    sender().connect(&receiver());
    rma_send(sender().ep(), receiver().worker());
    sender().flush_worker();
}

UCS_TEST_P(test_ucp_wireup_rma, two_sided_wireup) {
    sender().connect(&receiver());
    if (&sender() != &receiver()) {
        receiver().connect(&sender());
    }

    rma_send(sender().ep(), receiver().worker());
    sender().flush_worker();
    rma_send(receiver().ep(), sender().worker());
    receiver().flush_worker();
}

UCS_TEST_P(test_ucp_wireup_rma, connect_disconnect) {
    sender().connect(&receiver());
    if (&sender() != &receiver()) {
        receiver().connect(&sender());
    }
    sender().disconnect();
    receiver().disconnect();
}

UCS_TEST_P(test_ucp_wireup_rma, disconnect_nonexistent) {
    sender().connect(&receiver());
    receiver().destroy_worker();
    sender().disconnect();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_tag)
UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup_rma)
