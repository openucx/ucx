/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucs/debug/debug.h>
}


ucp_params_t test_ucp_tag::get_ctx_params() {
    ucp_params_t params = ucp_test::get_ctx_params();
    params.features     = UCP_FEATURE_TAG;
    params.request_size = sizeof(request);
    params.request_init = request_init;
    return params;
}

void test_ucp_tag::init()
{
    ucp_test::init();
    sender   = create_entity();
    receiver = create_entity();
    sender->connect(receiver);

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;
}

void test_ucp_tag::cleanup()
{
    sender->flush();
    receiver->flush();
    sender->disconnect();
    ucp_test::cleanup();
}

void test_ucp_tag::request_init(void *request)
{
    struct request *req = (struct request *)request;
    req->completed       = false;
    req->info.length     = 0;
    req->info.sender_tag = 0;
}

void test_ucp_tag::request_release(struct request *req)
{
    req->completed = false;
    ucp_request_release(req);
}

void test_ucp_tag::send_callback(void *request, ucs_status_t status)
{
    struct request *req = (struct request *)request;
    ucs_assert(req->completed == false);
    req->status    = status;
    req->completed = true;
}

void test_ucp_tag::recv_callback(void *request, ucs_status_t status,
                                 ucp_tag_recv_info_t *info)
{
    struct request *req = (struct request *)request;
    ucs_assert(req->completed == false);
    req->status    = status;
    req->completed = true;
    if (status == UCS_OK) {
        req->info      = *info;
    }
}

void test_ucp_tag::wait(request *req)
{
    while (!req->completed) {
        progress();
    }
}

void test_ucp_tag::send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
                          ucp_tag_t tag)
{
    request *req;
    req = (request*)ucp_tag_send_nb(sender->ep(), buffer, count, datatype,
                                    tag, send_callback);
    if (!UCS_PTR_IS_PTR(req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    } else {
        wait(req);
        request_release(req);
    }
}

ucs_status_t test_ucp_tag::recv_b(void *buffer, size_t count, ucp_datatype_t datatype,
                                  ucp_tag_t tag, ucp_tag_t tag_mask,
                                  ucp_tag_recv_info_t *info)
{
    ucs_status_t status;
    request *req;

    req = (request*)ucp_tag_recv_nb(receiver->worker(), buffer, count, datatype,
                                    tag, tag_mask, recv_callback);
    if (UCS_PTR_IS_ERR(req)) {
        return UCS_PTR_STATUS(req);
    } else if (req == NULL) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
    } else {
        wait(req);
        status = req->status;
        *info  = req->info;
        request_release(req);
        return status;
    }
}

void* test_ucp_tag::dt_start(size_t count)
{
    dt_gen_state *dt_state = new dt_gen_state;
    dt_state->count   = count;
    dt_state->started = 1;
    dt_state->magic   = MAGIC;
    dt_gen_start_count++;
    return dt_state;
}

void* test_ucp_tag::dt_start_pack(void *context, const void *buffer, size_t count)
{
    return dt_start(count);
}

void* test_ucp_tag::dt_start_unpack(void *context, void *buffer, size_t count)
{
    return dt_start(count);
}

size_t test_ucp_tag::dt_packed_size(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    return dt_state->count * sizeof(uint32_t);
}

size_t test_ucp_tag::dt_pack(void *state, size_t offset, void *dest, size_t max_length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    uint32_t *p = (uint32_t*)dest;
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    ucs_assert((offset % sizeof(uint32_t)) == 0);

    count = ucs_min(max_length / sizeof(uint32_t),
                    dt_state->count - (offset / sizeof(uint32_t)));
    for (unsigned i = 0; i < count; ++i) {
        p[i] = (offset / sizeof(uint32_t)) + i;
    }
    return count * sizeof(uint32_t);
}

ucs_status_t test_ucp_tag::dt_unpack(void *state, size_t offset, const void *src,
                                     size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    count = length / sizeof(uint32_t);
    for (unsigned i = 0; i < count; ++i) {
        uint32_t expected = (offset / sizeof(uint32_t)) + i;
        uint32_t actual   = ((uint32_t*)src)[i];
        if (actual != expected) {
            UCS_TEST_ABORT("Invalid data at index " << i << ". expected: " <<
                           expected << " actual: " << actual << ".");
        }
    }
    return UCS_OK;
}

void test_ucp_tag::dt_finish(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    --dt_state->started;
    EXPECT_EQ(0, dt_state->started);
    dt_gen_finish_count++;
    delete dt_state;
}

const ucp_datatype_t test_ucp_tag::DATATYPE = ucp_dt_make_contig(1);

ucp_generic_dt_ops test_ucp_tag::test_dt_ops = {
    test_ucp_tag::dt_start_pack,
    test_ucp_tag::dt_start_unpack,
    test_ucp_tag::dt_packed_size,
    test_ucp_tag::dt_pack,
    test_ucp_tag::dt_unpack,
    test_ucp_tag::dt_finish
};

int test_ucp_tag::dt_gen_start_count = 0;
int test_ucp_tag::dt_gen_finish_count = 0;
