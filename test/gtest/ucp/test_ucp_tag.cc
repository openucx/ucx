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
    sender().connect(&receiver());

    ucp_context_query(receiver().ucph(), &ctx_attr);

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;
}

void test_ucp_tag::request_init(void *request)
{
    struct request *req = (struct request *)request;
    req->completed       = false;
    req->external        = false;
    req->info.length     = 0;
    req->info.sender_tag = 0;
}

void test_ucp_tag::request_release(struct request *req)
{
    req->completed = false;

    if (req->external) {
        free(req->req_mem);
    } else {
        ucp_request_release(req);
    }
}

test_ucp_tag::request* test_ucp_tag::request_alloc()
{
    void *mem = malloc(ctx_attr.request_size + sizeof(request));
    request *req = (request*)((char*)mem + ctx_attr.request_size);
    request_init(req);
    req->external = true;
    req->req_mem = mem;
    return req;
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
    if (GetParam().variant == RECV_REQ_EXTERNAL) {
        ucp_tag_recv_info_t recv_info;
        ucs_status_t status = ucp_request_test(req, &recv_info);

        while (status == UCS_INPROGRESS) {
            progress();
            status =  ucp_request_test(req, &recv_info);
        }
        if (req->external) {
            recv_callback(req, status, &recv_info);
        }
    } else {
        while (!req->completed) {
            progress();
        }
    }
}

test_ucp_tag::request *
test_ucp_tag::send_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_tag_t tag)
{
    request *req;
    req = (request*)ucp_tag_send_nb(sender().ep(), buffer, count, datatype,
                                    tag, send_callback);
    if (UCS_PTR_IS_ERR(req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    }
    return req;
}

void test_ucp_tag::send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
                          ucp_tag_t tag)
{
    request *req = send_nb(buffer, count, datatype, tag);
    if (req != NULL) {
        wait(req);
        request_release(req);
    }
}

test_ucp_tag::request *
test_ucp_tag::send_sync_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                           ucp_tag_t tag)
{
    request *req;
    req = (request*)ucp_tag_send_sync_nb(sender().ep(), buffer, count, datatype,
                                         tag, send_callback);
    if (!UCS_PTR_IS_PTR(req)) {
        UCS_TEST_ABORT("ucp_tag_send_sync_nb returned status " <<
                       ucs_status_string(UCS_PTR_STATUS(req)));
    } else {
        return req;
    }
}

test_ucp_tag::request*
test_ucp_tag::recv_nb(void *buffer, size_t count, ucp_datatype_t dt,
                      ucp_tag_t tag, ucp_tag_t tag_mask)
{
    return (GetParam().variant == RECV_REQ_EXTERNAL) ?
                    recv_req_nb(buffer, count, dt, tag, tag_mask) :
                    recv_cb_nb(buffer, count, dt, tag, tag_mask);
}

test_ucp_tag::request*
test_ucp_tag::recv_req_nb(void *buffer, size_t count, ucp_datatype_t dt,
                          ucp_tag_t tag, ucp_tag_t tag_mask)
{
    request *req = request_alloc();

    ucs_status_t status = ucp_tag_recv_nbr(receiver().worker(), buffer, count,
                                           dt, tag, tag_mask, req);
    if (status != UCS_OK && status != UCS_INPROGRESS) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned status " <<
                       ucs_status_string(UCS_PTR_STATUS(req)));
    }
    return req;
}

test_ucp_tag::request*
test_ucp_tag::recv_cb_nb(void *buffer, size_t count, ucp_datatype_t dt,
                         ucp_tag_t tag, ucp_tag_t tag_mask)
{

    request *req = (request*) ucp_tag_recv_nb(receiver().worker(), buffer, count,
                                              dt, tag, tag_mask, recv_callback);
    if (UCS_PTR_IS_ERR(req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    } else if (req == NULL) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
    }
    return req;
}

ucs_status_t
test_ucp_tag::recv_b(void *buffer, size_t count, ucp_datatype_t dt, ucp_tag_t tag,
                     ucp_tag_t tag_mask, ucp_tag_recv_info_t *info)
{
    return (GetParam().variant == RECV_REQ_EXTERNAL) ?
                    recv_req_b(buffer, count, dt, tag, tag_mask, info) :
                    recv_cb_b(buffer, count, dt, tag, tag_mask, info);
}

ucs_status_t test_ucp_tag::recv_cb_b(void *buffer, size_t count, ucp_datatype_t datatype,
                                     ucp_tag_t tag, ucp_tag_t tag_mask,
                                     ucp_tag_recv_info_t *info)
{
    ucs_status_t status;
    request *req;

    req = (request*)ucp_tag_recv_nb(receiver().worker(), buffer, count, datatype,
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

ucs_status_t test_ucp_tag::recv_req_b(void *buffer, size_t count, ucp_datatype_t datatype,
                                      ucp_tag_t tag, ucp_tag_t tag_mask,
                                      ucp_tag_recv_info_t *info)
{
    request *req = request_alloc();
    ucs_status_t status = ucp_tag_recv_nbr(receiver().worker(), buffer, count,
                                           datatype, tag, tag_mask, req);
    if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
        wait(req);
        status = req->status;
        *info  = req->info;
    }
    request_release(req);
    return status;
}

void* test_ucp_tag::dt_common_start(size_t count)
{
    dt_gen_state *dt_state = new dt_gen_state;
    dt_state->count   = count;
    dt_state->started = 1;
    dt_state->magic   = MAGIC;
    dt_gen_start_count++;
    return dt_state;
}

void* test_ucp_tag::dt_common_start_pack(void *context, const void *buffer, size_t count)
{
    return dt_common_start(count);
}

void* test_ucp_tag::dt_common_start_unpack(void *context, void *buffer, size_t count)
{
    return dt_common_start(count);
}

size_t test_ucp_tag::dt_uint32_packed_size(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    return dt_state->count * sizeof(uint32_t);
}

size_t test_ucp_tag::dt_uint32_pack(void *state, size_t offset, void *dest, size_t max_length)
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

ucs_status_t test_ucp_tag::dt_uint32_unpack(void *state, size_t offset, const void *src,
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
                           expected << " actual: " << actual << " offset: " <<
                           offset << ".");
        }
    }
    return UCS_OK;
}

void test_ucp_tag::dt_common_finish(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    --dt_state->started;
    EXPECT_EQ(0, dt_state->started);
    dt_gen_finish_count++;
    delete dt_state;
}

size_t test_ucp_tag::dt_uint8_packed_size(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    return dt_state->count;
}

size_t test_ucp_tag::dt_uint8_pack(void *state, size_t offset, void *dest, size_t max_length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    uint8_t *p = (uint8_t*)dest;
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    ucs_assert((offset % sizeof(uint8_t)) == 0);

    count = ucs_min(max_length, dt_state->count - offset );
    for (unsigned i = 0; i < count; ++i) {
        p[i] = offset + i;
    }
    return count * sizeof(uint8_t);
}

ucs_status_t test_ucp_tag::dt_uint8_unpack(void *state, size_t offset, const void *src,
                                           size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    for (unsigned i = 0; i < length; ++i) {
        uint8_t expected = offset + i;
        uint8_t actual   = ((uint8_t*)src)[i];
        if (actual != expected) {
            UCS_TEST_ABORT("Invalid data at index " << i << ". expected: " <<
                           expected << " actual: " << actual << " offset: " <<
                           offset << ".");
        }
    }
    return UCS_OK;
}

const ucp_datatype_t test_ucp_tag::DATATYPE     = ucp_dt_make_contig(1);
const ucp_datatype_t test_ucp_tag::DATATYPE_IOV = ucp_dt_make_iov();

ucp_generic_dt_ops test_ucp_tag::test_dt_uint32_ops = {
    test_ucp_tag::dt_common_start_pack,
    test_ucp_tag::dt_common_start_unpack,
    test_ucp_tag::dt_uint32_packed_size,
    test_ucp_tag::dt_uint32_pack,
    test_ucp_tag::dt_uint32_unpack,
    test_ucp_tag::dt_common_finish
};

ucp_generic_dt_ops test_ucp_tag::test_dt_uint8_ops = {
    test_ucp_tag::dt_common_start_pack,
    test_ucp_tag::dt_common_start_unpack,
    test_ucp_tag::dt_uint8_packed_size,
    test_ucp_tag::dt_uint8_pack,
    test_ucp_tag::dt_uint8_unpack,
    test_ucp_tag::dt_common_finish
};

int test_ucp_tag::dt_gen_start_count = 0;
int test_ucp_tag::dt_gen_finish_count = 0;
ucp_context_attr_t test_ucp_tag::ctx_attr;
