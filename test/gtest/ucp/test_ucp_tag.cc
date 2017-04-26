/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucp/core/ucp_context.h>
}


ucp_params_t test_ucp_tag::get_ctx_params() {
    ucp_params_t params = ucp_test::get_ctx_params();
    params.field_mask  |= UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_INIT |
                          UCP_PARAM_FIELD_REQUEST_SIZE;
    params.features     = UCP_FEATURE_TAG;
    params.request_size = sizeof(request);
    params.request_init = request_init;
    return params;
}

void test_ucp_tag::init()
{
    ucp_test::init();
    sender().connect(&receiver());

    ctx_attr.field_mask = 0;
    ctx_attr.field_mask |= UCP_ATTR_FIELD_REQUEST_SIZE;
    ctx_attr.field_mask |= UCP_ATTR_FIELD_THREAD_MODE;
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
    if (req->external) {
        free(req->req_mem);
    } else {
        req->completed = false;
        ucp_request_release(req);
    }
}

void test_ucp_tag::request_free(struct request *req)
{
    if (req->external) {
        free(req->req_mem);
    } else {
        req->completed = false;
        ucp_request_free(req);
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

void test_ucp_tag::wait_and_validate(request *req)
{
    if (req == NULL) {
        return;
    }

    wait(req);
    EXPECT_TRUE(req->completed);
    EXPECT_EQ(UCS_OK, req->status);
    request_release(req);
}

void test_ucp_tag::wait_for_unexpected_msg(ucp_context_h context, double sec)
{
    /* Wait for some message to be added to unexpected queue */
    ucs_time_t timeout = ucs_get_time() + ucs_time_from_sec(sec);

    do {
        short_progress_loop();
    } while (ucp_tag_unexp_is_empty(&context->tm) && (ucs_get_time() < timeout));
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

template <typename T>
size_t test_ucp_tag::dt_packed_size(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    return dt_state->count * sizeof(T);
}

template <typename T>
size_t test_ucp_tag::dt_pack(void *state, size_t offset, void *dest, size_t max_length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    T *p = reinterpret_cast<T*> (dest);
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    ucs_assert((offset % sizeof(T)) == 0);

    count = ucs_min(max_length / sizeof(T),
                    dt_state->count - (offset / sizeof(T)));
    for (unsigned i = 0; i < count; ++i) {
        p[i] = (offset / sizeof(T)) + i;
    }
    return count * sizeof(T);
}

template <typename T>
ucs_status_t test_ucp_tag::dt_unpack(void *state, size_t offset, const void *src,
                                     size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    uint32_t count;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    count = length / sizeof(T);
    for (unsigned i = 0; i < count; ++i) {
        T expected = (offset / sizeof(T)) + i;
        T actual   = ((T*)src)[i];
        if (actual != expected) {
            UCS_TEST_ABORT("Invalid data at index " << i << ". expected: " <<
                           expected << " actual: " << actual << " offset: " <<
                           offset << ".");
        }
    }
    return UCS_OK;
}

ucs_status_t test_ucp_tag::dt_err_unpack(void *state, size_t offset, const void *src,
                                         size_t length)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;

    EXPECT_GT(dt_gen_start_count, dt_gen_finish_count);
    EXPECT_EQ(1, dt_state->started);
    EXPECT_EQ(uint32_t(MAGIC), dt_state->magic);

    return UCS_ERR_NO_MEMORY;
}

void test_ucp_tag::dt_common_finish(void *state)
{
    dt_gen_state *dt_state = (dt_gen_state*)state;
    --dt_state->started;
    EXPECT_EQ(0, dt_state->started);
    dt_gen_finish_count++;
    delete dt_state;
}

const ucp_datatype_t test_ucp_tag::DATATYPE     = ucp_dt_make_contig(1);
const ucp_datatype_t test_ucp_tag::DATATYPE_IOV = ucp_dt_make_iov();

ucp_generic_dt_ops test_ucp_tag::test_dt_uint32_ops = {
    test_ucp_tag::dt_common_start_pack,
    test_ucp_tag::dt_common_start_unpack,
    test_ucp_tag::dt_packed_size<uint32_t>,
    test_ucp_tag::dt_pack<uint32_t>,
    test_ucp_tag::dt_unpack<uint32_t>,
    test_ucp_tag::dt_common_finish
};

ucp_generic_dt_ops test_ucp_tag::test_dt_uint8_ops = {
    test_ucp_tag::dt_common_start_pack,
    test_ucp_tag::dt_common_start_unpack,
    test_ucp_tag::dt_packed_size<uint8_t>,
    test_ucp_tag::dt_pack<uint8_t>,
    test_ucp_tag::dt_unpack<uint8_t>,
    test_ucp_tag::dt_common_finish
};

ucp_generic_dt_ops test_ucp_tag::test_dt_uint32_err_ops = {
    test_ucp_tag::dt_common_start_pack,
    test_ucp_tag::dt_common_start_unpack,
    test_ucp_tag::dt_packed_size<uint32_t>,
    test_ucp_tag::dt_pack<uint32_t>,
    test_ucp_tag::dt_err_unpack,
    test_ucp_tag::dt_common_finish
};

int test_ucp_tag::dt_gen_start_count = 0;
int test_ucp_tag::dt_gen_finish_count = 0;
ucp_context_attr_t test_ucp_tag::ctx_attr;
