/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "test_ucp_tag.h"
#include "ucp_datatype.h"

extern "C" {
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
}

#include <sys/mman.h>
#include <vector>


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
    sender().connect(&receiver(), get_ep_params());

    ctx_attr.field_mask = 0;
    ctx_attr.field_mask |= UCP_ATTR_FIELD_REQUEST_SIZE;
    ctx_attr.field_mask |= UCP_ATTR_FIELD_THREAD_MODE;
    ucp_context_query(receiver().ucph(), &ctx_attr);

    ucp::dt_gen_start_count  = 0;
    ucp::dt_gen_finish_count = 0;
}

void test_ucp_tag::enable_tag_mp_offload()
{
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_ENABLE", "y"));
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_MP_SRQ_ENABLE", "try"));
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_MP_NUM_STRIDES", "8"));
    m_env.push_back(new ucs::scoped_setenv("UCX_IB_MLX5_DEVX_OBJECTS",
                                           "dct,dcsrq,rcsrq,rcqp"));
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

void test_ucp_tag::wait(request *req, int buf_index)
{
    int worker_index = get_worker_index(buf_index);

    if (is_external_request()) {
        ucp_tag_recv_info_t tag_info;
        ucs_status_t        status = ucp_request_test(req, &tag_info);

        while (status == UCS_INPROGRESS) {
            progress(worker_index);
            status = ucp_request_test(req, &tag_info);
        }
        if (req->external) {
            recv_callback(req, status, &tag_info);
        }
    } else {
        while (!req->completed) {
            progress(worker_index);
            if ((req->external) &&
                (ucp_request_check_status(req) == UCS_OK)) {
                return;
            }
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

void test_ucp_tag::wait_for_unexpected_msg(ucp_worker_h worker, double sec)
{
    /* Wait for some message to be added to unexpected queue */
    ucs_time_t timeout = ucs_get_time() + ucs_time_from_sec(sec);

    do {
        short_progress_loop();
    } while (ucp_tag_unexp_is_empty(&worker->tm) && (ucs_get_time() < timeout));
}

void test_ucp_tag::check_offload_support(bool offload_required)
{
    bool offload_supported = ucp_ep_is_tag_offload_enabled(ucp_ep_config(sender().ep()));
    if (offload_supported != offload_required) {
        cleanup();
        std::string reason = offload_supported ? "tag offload" : "no tag offload";
        UCS_TEST_SKIP_R(reason);
    }
}

int test_ucp_tag::get_worker_index(int buf_index)
{
    int worker_index = 0;
    if (GetParam().thread_type == MULTI_THREAD_CONTEXT) {
        worker_index = buf_index;
    } else if (GetParam().thread_type == SINGLE_THREAD) {
        ucs_assert((buf_index == 0) && (worker_index == 0));
    }
    return worker_index;
}

test_ucp_tag::request *
test_ucp_tag::send(entity &sender, send_type_t type, const void *buffer,
                   size_t count, ucp_datatype_t datatype, ucp_tag_t tag,
                   int buf_index)
{
    int worker_index = get_worker_index(buf_index);
    request *req;
    ucs_status_t status;

    switch (type) {
    case SEND_B:
    case SEND_NB:
        req = (request*)ucp_tag_send_nb(sender.ep(worker_index), buffer, count,
                                        datatype, tag, send_callback);
        if ((req != NULL) && (type == SEND_B)) {
            wait(req, get_worker_index(buf_index));
            request_release(req);
            return NULL;
        }

        if (UCS_PTR_IS_ERR(req)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(req));
        }
        break;
    case SEND_NBR:
        req = request_alloc();
        status = ucp_tag_send_nbr(sender.ep(worker_index), buffer, count,
                                  datatype, tag, req);
        ASSERT_UCS_OK_OR_INPROGRESS(status);
        if (status == UCS_OK) {
            request_free(req);
            return (request*)UCS_STATUS_PTR(UCS_OK);
        }
        break;
    case SEND_SYNC_NB:
        return (request*)ucp_tag_send_sync_nb(sender.ep(worker_index), buffer,
                                              count, datatype, tag, send_callback);
    default:
        return NULL;
    }

    return req;
}

test_ucp_tag::request *
test_ucp_tag::send_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_tag_t tag, int buf_index)
{
    return send(sender(), SEND_NB, buffer, count, datatype, tag, buf_index);
}

test_ucp_tag::request *
test_ucp_tag::send_nbr(const void *buffer, size_t count,
                       ucp_datatype_t datatype,
                       ucp_tag_t tag, int buf_index)
{
    return send(sender(), SEND_NBR, buffer, count, datatype, tag, buf_index);
}


void test_ucp_tag::send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
                          ucp_tag_t tag, int buf_index)
{
    send(sender(), SEND_B, buffer, count, datatype, tag, buf_index);
}

test_ucp_tag::request *
test_ucp_tag::send_sync_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                           ucp_tag_t tag, int buf_index)
{
    return send(sender(), SEND_SYNC_NB, buffer, count, datatype, tag, buf_index);
}

test_ucp_tag::request*
test_ucp_tag::recv(entity &receiver, recv_type_t type, void *buffer,
                   size_t count, ucp_datatype_t datatype,
                   ucp_tag_t tag, ucp_tag_t tag_mask,
                   ucp_tag_recv_info_t *info, int buf_index)
{
    int worker_index = get_worker_index(buf_index);
    request *req;
    ucs_status_t status;

    switch (type) {
    case RECV_B:
    case RECV_NB:
        req = (request*)ucp_tag_recv_nb(receiver.worker(worker_index), buffer, count,
                                        datatype, tag, tag_mask, recv_callback);
        if (type == RECV_NB) {
            if (UCS_PTR_IS_ERR(req)) {
                ASSERT_UCS_OK(UCS_PTR_STATUS(req));
            } else if (req == NULL) {
                UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
            }
        } else {
            if (UCS_PTR_IS_ERR(req)) {
                return req;
            } else if (req == NULL) {
                UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
            } else {
                wait(req, worker_index);
                status = req->status;
                *info  = req->info;
                request_release(req);
                return (request*)UCS_STATUS_PTR(status);
            }
        }
        break;
    case RECV_BR:
    case RECV_NBR:
        req = request_alloc();
        status = ucp_tag_recv_nbr(receiver.worker(worker_index), buffer,
                                  count, datatype, tag, tag_mask, req);
        if (type == RECV_NBR) {
            if (UCS_STATUS_IS_ERR(status)) {
                UCS_TEST_ABORT("ucp_tag_recv_nb returned status " <<
                               ucs_status_string(status));
            }
        } else {
            if (!UCS_STATUS_IS_ERR(status)) {
                wait(req, worker_index);
                status = req->status;
                *info  = req->info;
                request_release(req);
                return (request*)UCS_STATUS_PTR(status);
            }
        }
        break;
    default:
        return NULL;
    }

    return req;
}

test_ucp_tag::request*
test_ucp_tag::recv_nb(void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_tag_t tag, ucp_tag_t tag_mask, int buf_index)
{
    recv_type_t type = is_external_request() ? RECV_NBR : RECV_NB;
    return recv(receiver(), type, buffer, count, datatype,
                tag, tag_mask, NULL, buf_index);
}

ucs_status_t
test_ucp_tag::recv_b(void *buffer, size_t count, ucp_datatype_t datatype,
                     ucp_tag_t tag, ucp_tag_t tag_mask,
                     ucp_tag_recv_info_t *info, int buf_index)
{
    recv_type_t type = is_external_request() ? RECV_BR : RECV_B;
    request* req = recv(receiver(), type, buffer, count, datatype,
                        tag, tag_mask, info, buf_index);
    return UCS_PTR_STATUS(req);
}

bool test_ucp_tag::is_external_request()
{
    return false;
}

ucp_context_attr_t test_ucp_tag::ctx_attr;


class test_ucp_tag_limits : public test_ucp_tag {
public:
    test_ucp_tag_limits() {
        m_test_offload = GetParam().variant;
        m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_ENABLE",
                                               ucs::to_string(m_test_offload).c_str()));
    }

    void init() {
        test_ucp_tag::init();
        check_offload_support(m_test_offload);
    }

    std::vector<ucp_test_param>
    static enum_test_params(const ucp_params_t& ctx_params,
                            const std::string& name,
                            const std::string& test_case_name,
                            const std::string& tls)
    {
        std::vector<ucp_test_param> result;
        generate_test_params_variant(ctx_params, name, test_case_name,
                                     tls, false, result);
        generate_test_params_variant(ctx_params, name, test_case_name + "/offload",
                                     tls, true, result);
        return result;
    }

protected:
    bool m_test_offload;
};

UCS_TEST_P(test_ucp_tag_limits, check_max_short_rndv_thresh_zero, "RNDV_THRESH=0") {
    size_t max_short =
        static_cast<size_t>(ucp_ep_config(sender().ep())->tag.eager.max_short + 1);

    // (maximal short + 1) <= RNDV thresh
    EXPECT_LE(max_short,
              ucp_ep_config(sender().ep())->tag.rndv.am_thresh.remote);
    EXPECT_LE(max_short,
              ucp_ep_config(sender().ep())->tag.rndv.rma_thresh.remote);

    // (maximal short + 1) <= RNDV fast local compl thresh
    EXPECT_LE(max_short,
              ucp_ep_config(sender().ep())->tag.rndv.am_thresh.local);
    EXPECT_LE(max_short,
              ucp_ep_config(sender().ep())->tag.rndv.rma_thresh.local);

    if (m_test_offload) {
        // There is a lower bound for rndv threshold with tag offload. We should
        // not send messages smaller than SW RNDV request size, because receiver
        // may temporarily store this request in the user buffer (which will
        // result in crash if the request does not fit user buffer).
        size_t min_rndv = ucp_ep_tag_offload_min_rndv_thresh(ucp_ep_config(sender().ep()));

        EXPECT_GT(min_rndv, 0ul); // min_rndv should be RTS size at least
        EXPECT_GE(min_rndv,
                  ucp_ep_config(sender().ep())->tag.rndv.am_thresh.local);
        EXPECT_GE(min_rndv,
                  ucp_ep_config(sender().ep())->tag.rndv.rma_thresh.local);
    }
}

UCS_TEST_P(test_ucp_tag_limits, check_max_short_zcopy_thresh_zero, "ZCOPY_THRESH=0") {
    size_t max_short =
        static_cast<size_t>(ucp_ep_config(sender().ep())->tag.eager.max_short + 1);

    // (maximal short + 1) <= ZCOPY thresh
    EXPECT_LE(max_short,
              ucp_ep_config(sender().ep())->tag.eager.zcopy_thresh[0]);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_limits)


class test_ucp_tag_fallback : public ucp_test {
public:
    void init() {
        /* forbid zcopy access because it will always fail due to read-only
         * memory pages (will fail to register memory) */
        modify_config("ZCOPY_THRESH", "inf");
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_TAG;
        return params;
    }

protected:
    static const size_t MSG_SIZE;
};

const size_t test_ucp_tag_fallback::MSG_SIZE  = 4 * 1024 * ucs_get_page_size();

UCS_TEST_P(test_ucp_tag_fallback, fallback)
{
    ucp_request_param_t param = {0};

    /* allocate read-only pages - it force ibv_reg_mr() failure */
    void *send_buffer = mmap(NULL, MSG_SIZE, PROT_READ,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(send_buffer, MAP_FAILED);

    std::vector<char> recv_buffer(MSG_SIZE);

    ucs_status_ptr_t recv_req = ucp_tag_recv_nbx(receiver().worker(),
                                                 &recv_buffer[0], MSG_SIZE,
                                                 0, 0, &param);
    ASSERT_UCS_PTR_OK(recv_req);

    ucs_status_ptr_t send_req = ucp_tag_send_nbx(sender().ep(), send_buffer,
                                                 MSG_SIZE, 0, &param);
    ASSERT_UCS_PTR_OK(send_req);

    wait(send_req);
    wait(recv_req);

    munmap(send_buffer, MSG_SIZE);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_fallback)
