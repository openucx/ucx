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


class test_ucp_wireup : public ucp_test {
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

void test_ucp_wireup::tag_send(ucp_ep_h from, ucp_worker_h to, int count)
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

void test_ucp_wireup::send_completion(void *request, ucs_status_t status)
{
}

void test_ucp_wireup::recv_completion(void *request, ucs_status_t status,
                                      ucp_tag_recv_info_t *info)
{
}

void test_ucp_wireup::wait(void *req)
{
    do {
        progress();
    } while (!ucp_request_is_completed(req));
    ucp_request_release(req);
}

UCS_TEST_P(test_ucp_wireup, address) {
    ucs_status_t status;
    size_t size;
    void *buffer;
    unsigned order[UCP_MAX_RESOURCES];

    entity *ent1 = create_entity();
    status = ucp_address_pack(ent1->worker(), NULL, -1, order, &size, &buffer);
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
    EXPECT_EQ(ent1->worker()->uuid, uuid);
    EXPECT_EQ(std::string(ucp_worker_get_name(ent1->worker())), std::string(name));
    EXPECT_LE(address_count, ent1->ucph()->num_tls);

    /* TODO test addresses */

    ucs_free(address_list);
    ucs_free(buffer);
}

UCS_TEST_P(test_ucp_wireup, empty_address) {
    ucs_status_t status;
    size_t size;
    void *buffer;
    unsigned order[UCP_MAX_RESOURCES];

    entity *ent1 = create_entity();
    status = ucp_address_pack(ent1->worker(), NULL, 0, order, &size, &buffer);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(buffer != NULL);
    ASSERT_GT(size, 0ul);

    char name[UCP_WORKER_NAME_MAX];
    uint64_t uuid;
    unsigned address_count;
    ucp_address_entry_t *address_list;

    ucp_address_unpack(buffer, &uuid, name, sizeof(name), &address_count,
                       &address_list);
    EXPECT_EQ(ent1->worker()->uuid, uuid);
    EXPECT_EQ(std::string(ucp_worker_get_name(ent1->worker())), std::string(name));
    EXPECT_LE(address_count, ent1->ucph()->num_tls);
    EXPECT_EQ(0u, address_count);

    ucs_free(address_list);
    ucs_free(buffer);
}


UCS_TEST_P(test_ucp_wireup, one_sided_wireup) {
    entity *ent1 = create_entity();
    entity *ent2 = create_entity();

    ent1->connect(ent2);
    tag_send(ent1->ep(), ent2->worker());
    ent1->flush_worker();
}

UCS_TEST_P(test_ucp_wireup, two_sided_wireup) {
    entity *ent1 = create_entity();
    entity *ent2 = create_entity();

    ent1->connect(ent2);
    ent2->connect(ent1);

    tag_send(ent1->ep(), ent2->worker());
    ent1->flush_worker();
    tag_send(ent2->ep(), ent1->worker());
    ent2->flush_worker();
}

UCS_TEST_P(test_ucp_wireup, reply_ep_send_before) {
    entity *ent1 = create_entity();
    entity *ent2 = create_entity();

    ent1->connect(ent2);

    ucp_ep_connect_remote(ent1->ep());

    /* Send a reply */
    ucp_ep_h ep = ucp_worker_get_reply_ep(ent2->worker(), ent1->worker()->uuid);
    tag_send(ep, ent1->worker());
    ent1->flush_worker();

    ucp_ep_destroy(ep);
}

UCS_TEST_P(test_ucp_wireup, reply_ep_send_after) {
    entity *ent1 = create_entity();
    entity *ent2 = create_entity();

    ent1->connect(ent2);

    ucp_ep_connect_remote(ent1->ep());

    /* Make sure the wireup message arrives before sending a reply */
    tag_send(ent1->ep(), ent2->worker());
    ent1->flush_worker();

    /* Send a reply */
    ucp_ep_h ep = ucp_worker_get_reply_ep(ent2->worker(), ent1->worker()->uuid);
    tag_send(ep, ent1->worker());
    ent1->flush_worker();

    ucp_ep_destroy(ep);
}

UCS_TEST_P(test_ucp_wireup, stress_connect) {
    entity *ent1 = create_entity();
    entity *ent2 = create_entity();

    for (int i = 0; i < 30; ++i) {
        ent1->connect(ent2);
        tag_send(ent1->ep(), ent2->worker(), 10000 / ucs::test_time_multiplier());
        ent2->connect(ent1);
        ent1->disconnect();
        ent2->disconnect();
    }
}

UCS_TEST_P(test_ucp_wireup, stress_connect2) {
    entity *ent1 = create_entity();
    entity *ent2 = create_entity();

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        ent1->connect(ent2);
        tag_send(ent1->ep(), ent2->worker(), 1);
        ent2->connect(ent1);
        ent1->disconnect();
        ent2->disconnect();
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_wireup)
