/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
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
#include <ucs/arch/atomic.h>
#include <ucs/memory/rcache.h>
#include <ucs/memory/rcache_int.h>
#include <ucp/proto/proto_select.inl>
}

#include <sys/mman.h>
#include <vector>


ucp_params_t test_ucp_tag::get_ctx_params() {
    ucp_params_t params = {};
    params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_INIT |
                          UCP_PARAM_FIELD_REQUEST_SIZE;
    params.features     = UCP_FEATURE_TAG;
    params.request_size = sizeof(request);
    params.request_init = request_init;
    return params;
}

void test_ucp_tag::get_test_variants(std::vector<ucp_test_variant>& variants)
{
    add_variant(variants, get_ctx_params());
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
}

void test_ucp_tag::request_init(void *request)
{
    struct request *req = (struct request *)request;
    req->completed       = false;
    req->external        = false;
    req->info.length     = 0;
    req->info.sender_tag = 0;
}

void test_ucp_tag::request_free(request *req)
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

void test_ucp_tag::send_callback(void *request, ucs_status_t status,
                                 void *user_data)
{
    struct request *req = (struct request *)(user_data ?: request);
    ucs_assert(req->completed == false);
    req->status    = status;
    req->completed = true;
}

/* TODO: deprecated, remove after complete migration to new API */
void test_ucp_tag::send_callback(void *request, ucs_status_t status)
{
    send_callback(request, status, NULL);
}

void test_ucp_tag::recv_callback(void *request, ucs_status_t status,
                                 const ucp_tag_recv_info_t *info,
                                 void *user_data)
{
    struct request *req = (struct request *)(user_data ?: request);
    ucs_assert(req->completed == false);
    req->status    = status;
    req->completed = true;
    if (status == UCS_OK) {
        req->info  = *info;
    }
}

/* TODO: deprecated, remove after complete migration to new API */
void test_ucp_tag::recv_callback(void *request, ucs_status_t status,
                                 ucp_tag_recv_info_t *info)
{
    recv_callback(request, status, info, NULL);
}

void test_ucp_tag::wait(void *ucx_req, void *user_data, int buf_index)
{
    request *req     = (request*)(user_data ?: ucx_req);
    int worker_index = get_worker_index(buf_index);

    if (is_external_request()) {
        ucp_tag_recv_info_t tag_info;
        ucs_status_t        status = ucp_request_test(ucx_req, &tag_info);

        while (status == UCS_INPROGRESS) {
            progress(worker_index);
            status = ucp_request_test(ucx_req, &tag_info);
        }
        if (req->external) {
            recv_callback(ucx_req, status, &tag_info, user_data);
        }
    } else {
        /* TODO: wait using request status only */
        while (!req->completed) {
            progress(worker_index);
            if ((req->external) &&
                (ucp_request_check_status(ucx_req) == UCS_OK)) {
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
    request_free(req);
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
    bool offload_supported = ucp_ep_config_key_has_tag_lane(
                               &ucp_ep_config(sender().ep())->key);
    if (offload_supported != offload_required) {
        cleanup();
        std::string reason = offload_supported ? "tag offload" : "no tag offload";
        UCS_TEST_SKIP_R(reason);
    }
}

int test_ucp_tag::get_worker_index(int buf_index)
{
    int worker_index = 0;
    if (get_variant_thread_type() == MULTI_THREAD_CONTEXT) {
        worker_index = buf_index;
    } else if (get_variant_thread_type() == SINGLE_THREAD) {
        ucs_assert((buf_index == 0) && (worker_index == 0));
    }
    return worker_index;
}

test_ucp_tag::request *
test_ucp_tag::send(entity &sender, send_type_t type, const void *buffer,
                   size_t count, ucp_datatype_t datatype, ucp_tag_t tag,
                   void *user_data, int buf_index)
{
    int worker_index = get_worker_index(buf_index);
    request *req;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype     = datatype;

    switch (type) {
    case SEND_B:
        param.op_attr_mask |= UCP_OP_ATTR_FLAG_FAST_CMPL;
        /* fallthrough */
    case SEND_NB:
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA;
        param.cb.send       = send_callback;
        param.user_data     = user_data;
        req                 = (request*)ucp_tag_send_nbx(sender.ep(worker_index),
                                                         buffer, count,
                                                         tag, &param);
        if ((req != NULL) && (type == SEND_B)) {
            wait(req, user_data, get_worker_index(buf_index));
            request_free(req);
            return NULL;
        }

        if (UCS_PTR_IS_ERR(req)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(req));
        }
        break;
    case SEND_NBR:
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_REQUEST;
        param.request       = request_alloc();
        req                 = (request*)ucp_tag_send_nbx(sender.ep(worker_index),
                                                         buffer, count, tag,
                                                         &param);
        if (req == NULL) {
            request_free((request*)param.request);
            return (request*)UCS_STATUS_PTR(UCS_OK);
        }

        if (user_data) {
            ((request*)user_data)->external = true;
        }

        break;
    case SEND_SYNC_NB:
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_CALLBACK;
        param.cb.send       = send_callback;
        return (request*)ucp_tag_send_sync_nbx(sender.ep(worker_index), buffer,
                                               count, tag, &param);
    default:
        return NULL;
    }

    return req;
}

test_ucp_tag::request *
test_ucp_tag::send_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_tag_t tag, void *user_data, int buf_index)
{
    return send(sender(), SEND_NB, buffer, count, datatype, tag, user_data,
                buf_index);
}

test_ucp_tag::request *
test_ucp_tag::send_nbr(const void *buffer, size_t count,
                       ucp_datatype_t datatype,
                       ucp_tag_t tag, void *user_data, int buf_index)
{
    return send(sender(), SEND_NBR, buffer, count, datatype, tag, user_data,
                buf_index);
}


void test_ucp_tag::send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
                          ucp_tag_t tag, void *user_data, int buf_index)
{
    send(sender(), SEND_B, buffer, count, datatype, tag, user_data, buf_index);
}

test_ucp_tag::request *
test_ucp_tag::send_sync_nb(const void *buffer, size_t count, ucp_datatype_t datatype,
                           ucp_tag_t tag, void *user_data, int buf_index)
{
    return send(sender(), SEND_SYNC_NB, buffer, count, datatype, tag, user_data,
                buf_index);
}

test_ucp_tag::request*
test_ucp_tag::recv(entity &receiver, recv_type_t type, void *buffer,
                   size_t count, ucp_datatype_t datatype,
                   ucp_tag_t tag, ucp_tag_t tag_mask,
                   ucp_tag_recv_info_t *info, void *user_data, int buf_index)
{
    int worker_index = get_worker_index(buf_index);
    request *req;
    ucs_status_t status;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    param.datatype     = datatype;

    switch (type) {
    case RECV_IMM:
        param.op_attr_mask      &= ~UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
        param.op_attr_mask      |= UCP_OP_ATTR_FIELD_RECV_INFO;
        param.recv_info.tag_info = info;
        // Fallthrough
    case RECV_B:
    case RECV_NB:
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA;
        param.cb.recv       = recv_callback;
        param.user_data     = user_data;
        req                 = (request*)ucp_tag_recv_nbx(receiver.worker(worker_index),
                                                         buffer, count, tag, tag_mask,
                                                         &param);
        if (type == RECV_IMM) {
            if (req != NULL) {
                UCS_TEST_ABORT("ucp_tag_recv_nbx returned non-NULL");
            }
        } else if (type == RECV_NB) {
            if (UCS_PTR_IS_ERR(req)) {
                ASSERT_UCS_OK(UCS_PTR_STATUS(req));
            } else if (req == NULL) {
                UCS_TEST_ABORT("ucp_tag_recv_nbx returned NULL");
            }
        } else {
            if (UCS_PTR_IS_ERR(req)) {
                return req;
            } else if (req == NULL) {
                UCS_TEST_ABORT("ucp_tag_recv_nbx returned NULL");
            } else {
                wait(req, user_data, worker_index);
                status = req->status;
                *info  = req->info;
                request_free(req);
                return (request*)UCS_STATUS_PTR(status);
            }
        }
        break;
    case RECV_BR:
    case RECV_NBR:
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_REQUEST;
        param.request       = request_alloc();
        req                 = (request*)ucp_tag_recv_nbx(receiver.worker(worker_index),
                                                         buffer, count, tag, tag_mask,
                                                         &param);
        if (type == RECV_NBR) {
            if (UCS_PTR_IS_ERR(req)) {
                UCS_TEST_ABORT("ucp_tag_recv_nbx returned status " <<
                               ucs_status_string(UCS_PTR_STATUS(req)));
            }
        } else {
            if (!UCS_PTR_IS_ERR(req)) {
                wait(req, NULL, worker_index);
                status = req->status;
                *info  = req->info;
                request_free(req);
                return (request*)UCS_STATUS_PTR(status);
            }
        }

        /* TODO: make refactoring of tests to add native user data processing */
        if (user_data) {
            ((request*)user_data)->external = true;
        }

        break;
    default:
        return NULL;
    }

    return req;
}

test_ucp_tag::request*
test_ucp_tag::recv_nb(void *buffer, size_t count, ucp_datatype_t datatype,
                      ucp_tag_t tag, ucp_tag_t tag_mask, void *user_data,
                      int buf_index)
{
    recv_type_t type = is_external_request() ? RECV_NBR : RECV_NB;
    return recv(receiver(), type, buffer, count, datatype,
                tag, tag_mask, NULL, user_data, buf_index);
}

ucs_status_t
test_ucp_tag::recv_b(void *buffer, size_t count, ucp_datatype_t datatype,
                     ucp_tag_t tag, ucp_tag_t tag_mask,
                     ucp_tag_recv_info_t *info, void *user_data, int buf_index)
{
    recv_type_t type = is_external_request() ? RECV_BR : RECV_B;
    request* req = recv(receiver(), type, buffer, count, datatype,
                        tag, tag_mask, info, user_data, buf_index);
    return UCS_PTR_STATUS(req);
}

ucs_status_t test_ucp_tag::recv_imm(void *buffer, size_t count,
                                    ucp_datatype_t datatype, ucp_tag_t tag,
                                    ucp_tag_t tag_mask,
                                    ucp_tag_recv_info_t *info, void *user_data,
                                    int buf_index)
{
    request *req = recv(receiver(), RECV_IMM, buffer, count, datatype, tag,
                        tag_mask, info, user_data, buf_index);
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
        m_test_offload = get_variant_value();
        if (m_test_offload) {
            m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_ENABLE", "y"));
        }
        m_tag_min_rndv = 0;
    }

    void init() {
        /* TODO: Currently all the tests are for intra-node communication only.
         * Find a way to create inter-node endpoint on a single node */
        test_ucp_tag::init();

        check_offload_support(m_test_offload);

        if (m_test_offload) {
            ucp_ep_config_t *cfg = ucp_ep_config(sender().ep());
            m_tag_min_rndv = ucp_ep_tag_offload_min_rndv_thresh(sender().ucph(),
                                                                &cfg->key);
        }
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        ucp_params_t params = get_ctx_params();
        params.features = UCP_FEATURE_TAG | UCP_FEATURE_AM;

        add_variant_with_value(variants, params, 0, "");
        add_variant_with_value(variants, params, 1, "offload");
    }

protected:
    bool    m_test_offload;
    size_t  m_tag_min_rndv;

    static void check_short_thresh(const ucp_memtype_thresh_t &thresh,
                                   size_t cfg_thresh, bool strict = false)
    {
        if (strict) {
            EXPECT_EQ(thresh.memtype_on + 1, cfg_thresh);
            EXPECT_EQ(thresh.memtype_off + 1, cfg_thresh);
        } else {
            EXPECT_LE(thresh.memtype_on + 1, cfg_thresh);
            EXPECT_LE(thresh.memtype_off + 1, cfg_thresh);
        }
    }

    void check_rndv_startup_config(size_t exp_rndv_intra_thresh,
                                   size_t exp_rndv_inter_thresh)
    {
        ucp_context_config_t *cfg = &sender().worker()->context->config.ext;

        EXPECT_EQ(exp_rndv_intra_thresh, cfg->rndv_intra_thresh);
        EXPECT_EQ(exp_rndv_inter_thresh, cfg->rndv_inter_thresh);
    }

    void check_tag_rndv_v2(size_t cfg_thresh)
    {
        ucp_ep_config_t *cfg = ucp_ep_config(sender().ep());

        if (m_test_offload) {
            /* If configured threshold is less than min_rndv, then expect exact
             * min_rndv limit for short messages */
            if (cfg_thresh < m_tag_min_rndv) {
                check_short_thresh(cfg->tag.offload.max_eager_short, m_tag_min_rndv,
                                   true);
            } else {
                check_short_thresh(cfg->tag.offload.max_eager_short, cfg_thresh);
            }
        } else {
            check_short_thresh(cfg->tag.max_eager_short, cfg_thresh);
        }
    }

    void check_am_rndv_v2(size_t cfg_thresh)
    {
        ucp_ep_config_t *cfg = ucp_ep_config(sender().ep());

        check_short_thresh(cfg->am_u.max_eager_short, cfg_thresh);
        check_short_thresh(cfg->am_u.max_reply_eager_short, cfg_thresh);
    }

    void check_ep_proto_rndv_v2(size_t cfg_thresh, bool expect_rndv)
    {
        ucp_ep_config_t *cfg = ucp_ep_config(sender().ep());
        const ucp_proto_config_t *proto_config;
        ucp_proto_select_elem_t value;

        /* Skip proto_select hash map check for HWTM since eager has certain
           max_frag threshold in that case and there is no reliable way
           on UCP side to obtain that value. So RNDV can be used instead 
           of eager even if RNDV_THRESH is set to higher value. */
        if (m_test_offload) {
            UCS_TEST_SKIP_R("Skip EP RNDV_THRESH check for HWTM");
        }

        kh_foreach_value(cfg->proto_select.hash, value, {
            /* Find index of the corresponding ucp_proto_threshold_elem_t
             * to handle the given message size */
            unsigned idx = 0;
            for (; cfg_thresh > value.thresholds[idx].max_msg_length; ++idx) {
                proto_config = &value.thresholds[idx].proto_config;
                /* Assert no rndv before expected limit */
                EXPECT_EQ(nullptr, strstr(proto_config->proto->name, "rndv"));
            }

            proto_config = &value.thresholds[idx].proto_config;
            if (expect_rndv) {
                EXPECT_NE(nullptr, strstr(proto_config->proto->name, "rndv"));
            } else {
                EXPECT_EQ(nullptr, strstr(proto_config->proto->name, "rndv"));
            }
        });
    }

    void check_rndv_threshold(size_t cfg_thresh)
    {
        ucp_context_config_t *cfg = &sender().worker()->context->config.ext;
        if (cfg->proto_enable) {
            /* Check proto_v2 rndv thresholds only when this protocol is
             * enabled, otherwise these checks are irrelevant */
            check_tag_rndv_v2(cfg_thresh);
            check_am_rndv_v2(cfg_thresh);
            check_ep_proto_rndv_v2(cfg_thresh, true);
        }
    }
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
        size_t min_rndv = ucp_ep_tag_offload_min_rndv_thresh(
                           sender().ucph(), &ucp_ep_config(sender().ep())->key);

        EXPECT_GT(min_rndv, 0ul); // min_rndv should be RTS size at least
        EXPECT_LE(min_rndv,
                  ucp_ep_config(sender().ep())->tag.rndv.am_thresh.local);
        EXPECT_LE(min_rndv,
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

UCS_TEST_P(test_ucp_tag_limits, check_rndv_thresh,
           "RNDV_THRESH=0")
{
    check_rndv_startup_config(0, 0);
    check_rndv_threshold(0);
}

UCS_TEST_P(test_ucp_tag_limits, check_rndv_intra_thresh,
           "RNDV_THRESH=auto,intra:20")
{
    check_rndv_startup_config(20, UCS_MEMUNITS_AUTO);
    check_rndv_threshold(20);
}

UCS_TEST_P(test_ucp_tag_limits, check_rndv_intra_thresh_large,
           "RNDV_THRESH=auto,intra:2000")
{
    check_rndv_startup_config(2000, UCS_MEMUNITS_AUTO);
    check_rndv_threshold(2000);
}

UCS_TEST_P(test_ucp_tag_limits, check_rndv_intra_thresh_inf,
           "RNDV_THRESH=auto,intra:inf")
{
    check_rndv_startup_config(UCS_MEMUNITS_INF, UCS_MEMUNITS_AUTO);
    /* check that rndv protocol is disabled */
    check_ep_proto_rndv_v2(UCS_MEMUNITS_INF, false);
}

UCS_TEST_P(test_ucp_tag_limits, check_rndv_intra_thresh_common,
           "RNDV_THRESH=10,intra:20")
{
    check_rndv_startup_config(20, 10);
    check_rndv_threshold(20);
}

UCS_TEST_P(test_ucp_tag_limits, check_rndv_intra_inter_thresh,
           "RNDV_THRESH=intra:20,inter:30")
{
    check_rndv_startup_config(20, 30);
    check_rndv_threshold(20);
}

UCS_TEST_P(test_ucp_tag_limits, check_rndv_inter_thresh,
           "RNDV_THRESH=auto,inter:30")
{
    check_rndv_startup_config(UCS_MEMUNITS_AUTO, 30);
    /* TODO: configure/mock inter-node in test */

    /* check that inter-node config is ignored for intra-node */
    check_ep_proto_rndv_v2(30, false);
}

UCS_TEST_P(test_ucp_tag_limits, check_rndv_inter_thresh_common,
           "RNDV_THRESH=1000,inter:30")
{
    check_rndv_startup_config(1000, 30);
    check_rndv_threshold(1000);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_limits)


class test_ucp_tag_nbx : public test_ucp_tag {
public:
    class completion_value {
    public:
        completion_value() : m_address(nullptr), m_value(0)
        {
        }

        completion_value(uint32_t *address, uint32_t value) :
            m_address(address), m_value(value)
        {
        }

        bool empty() const
        {
            return (m_address == nullptr) && (m_value == 0);
        }

        uint32_t *m_address;
        uint32_t  m_value;
    };

    test_ucp_tag_nbx()
    {
        if (disable_proto()) {
            modify_config("PROTO_ENABLE", "n");
        }
    }

    void init() {
        stats_activate();
        test_ucp_tag::init();
    }

    virtual void cleanup()
    {
        test_ucp_tag::cleanup();
        stats_restore();
    }

    static void
    get_test_variants_prereg(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_TAG, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_TAG, 1, "prereg");
    }

    static void get_test_variants_proto(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, get_test_variants_prereg, 0);
        if (!RUNNING_ON_VALGRIND) {
            add_variant_values(variants, get_test_variants_prereg, 1,
                               "proto_v1");
        }
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, get_test_variants_proto, 0);
        add_variant_values(variants, get_test_variants_proto, 1, "iov");
    }

protected:
    static const size_t MSG_SIZE;
    static const ucp_request_param_t null_param;
    static const completion_value null_completion;
    static const size_t iov_count;

    bool prereg() const
    {
        return get_variant_value(0);
    }

    bool disable_proto() const
    {
        return get_variant_value(1);
    }

    bool is_iov() const
    {
        return get_variant_value(2);
    }

    static void send_callback(void *req, ucs_status_t status,
                              void *user_data)
    {
        request_free((request*)req);
        ucs_atomic_add32((volatile uint32_t*)user_data, 1);
    }

    static void recv_callback(void *req, ucs_status_t status,
                              const ucp_tag_recv_info_t *info,
                              void *user_data)
    {
        request_free((request*)req);
        ucs_atomic_add32((volatile uint32_t*)user_data, 1);
    }

    void do_send_recv(const ucp::data_type_desc_t &send_dt,
                      ucp::data_type_desc_t &recv_dt,
                      const ucp_request_param_t &sparam,
                      const ucp_request_param_t &rparam)
    {
        const ssize_t recv_size = recv_dt.extent();
        ucs_assert(recv_size > 0);

        ucs_status_ptr_t recv_req = ucp_tag_recv_nbx(receiver().worker(),
                                                     recv_dt.buf(), recv_size,
                                                     0, 0, &rparam);
        ASSERT_UCS_PTR_OK(recv_req);

        const ssize_t send_size = send_dt.extent();
        ucs_assert(send_size > 0);
        ucs_status_ptr_t send_req = ucp_tag_send_nbx(sender().ep(),
                                                     send_dt.buf(), send_size,
                                                     0, &sparam);
        ASSERT_UCS_PTR_OK(send_req);

        if (!(rparam.op_attr_mask & UCP_OP_ATTR_FIELD_REQUEST)) {
            request_wait(recv_req);
        }

        if (!(sparam.op_attr_mask & UCP_OP_ATTR_FIELD_REQUEST)) {
            request_wait(send_req);
        }
    }

    void test_prereg_rcache_stats(const ucp::data_type_desc_t &send_dt,
                                  ucp::data_type_desc_t &recv_dt,
                                  const ucp_request_param_t &sparam,
                                  const ucp_request_param_t &rparam)
    {
        if (sparam.op_attr_mask & UCP_OP_ATTR_FIELD_REQUEST) {
            /* Can't send multiple requests for a single user request memory
               handle */
            return;
        }

        int prev_rcache_get_count;
        prev_rcache_get_count = UCS_STATS_GET_COUNTER(
                sender().ucph()->rcache->stats, UCS_RCACHE_GETS);

        for (int i = 0; i < 10; ++i) {
            do_send_recv(send_dt, recv_dt, sparam, rparam);
        }

        int rcache_get_count;
        rcache_get_count = UCS_STATS_GET_COUNTER(sender().ucph()->rcache->stats,
                                                 UCS_RCACHE_GETS);

        /* Compare counters before and after iterations */
        EXPECT_EQ(prev_rcache_get_count, rcache_get_count);
    }

    void test_recv_send(size_t size, const void *send_buffer, void *recv_buffer,
                        ucp_request_param_t send_param = null_param,
                        ucp_request_param_t recv_param = null_param,
                        completion_value completion = null_completion)
    {
        ucp_datatype_t datatype = is_iov() ? ucp_dt_make_iov() :
                                             ucp_dt_make_contig(1);
        ucp::data_type_desc_t send_dt(datatype, send_buffer, size, iov_count);

        /* Currently recv side supports only contig datatype */
        ucp::data_type_desc_t recv_dt(ucp_dt_make_contig(1), recv_buffer, size);

        send_param.datatype      = datatype;
        send_param.op_attr_mask |= UCP_OP_ATTR_FIELD_DATATYPE;

        if (prereg()) {
            send_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            recv_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            send_param.memh = sender().mem_map((void*)send_buffer, size);
            recv_param.memh = receiver().mem_map(recv_buffer, size);
        }

        do_send_recv(send_dt, recv_dt, send_param, recv_param);

        if (prereg() && !is_self() && (!is_iov() || is_proto_enabled())) {
            /* Not relevant for 'self' because both sender and receiver are the same entity.
               Must be called before request is freed by free_callback (wait_for_value). */
            /* User-provided memh on iov supported only with proto_v2 */
            test_prereg_rcache_stats(send_dt, recv_dt, send_param, recv_param);
        }

        if (!completion.empty()) {
            /* Wait for completion indication */
            wait_for_value(completion.m_address, completion.m_value);
        }

        if (prereg()) {
            sender().mem_unmap(send_param.memh);
            receiver().mem_unmap(recv_param.memh);
        }
    }

    void test_recv_send(size_t size = MSG_SIZE)
    {
        std::vector<char> send_buffer(size);
        std::vector<char> recv_buffer(size);
        test_recv_send(size, &send_buffer[0], &recv_buffer[0]);
    }
};

const size_t test_ucp_tag_nbx::MSG_SIZE  = 4 * UCS_KBYTE * ucs_get_page_size();
const ucp_request_param_t test_ucp_tag_nbx::null_param = {0};
const test_ucp_tag_nbx::completion_value test_ucp_tag_nbx::null_completion =
        {nullptr, 0};
const size_t test_ucp_tag_nbx::iov_count               = 20;

UCS_TEST_P(test_ucp_tag_nbx, basic)
{
    test_recv_send();
}

UCS_TEST_P(test_ucp_tag_nbx, eager_zcopy, "ZCOPY_THRESH=0", "RNDV_THRESH=inf")
{
    test_recv_send(4 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_tag_nbx, rndv_bcopy, "ZCOPY_THRESH=inf", "RNDV_THRESH=0")
{
    test_recv_send(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_tag_nbx, rndv_zcopy, "ZCOPY_THRESH=0", "RNDV_THRESH=0")
{
    test_recv_send(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_tag_nbx, fallback, "ZCOPY_THRESH=inf", "PROTO_ENABLE=n")
{
    if (!disable_proto() || prereg()) {
        UCS_TEST_SKIP_R(
                "protoV2/prereg are not supported for partial md reg failure");
    }

    /* allocate read-only pages - it force ibv_reg_mr() failure */
    void *send_buffer = mmap(NULL, MSG_SIZE, PROT_READ,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(send_buffer, MAP_FAILED);

    std::vector<char> recv_buffer(MSG_SIZE);

    test_recv_send(MSG_SIZE, send_buffer, &recv_buffer[0]);

    munmap(send_buffer, MSG_SIZE);
}

UCS_TEST_P(test_ucp_tag_nbx, external_request_free)
{
    ucp_request_param_t send_param = {0};
    ucp_request_param_t recv_param = {0};
    uint32_t completed             = 0;
    uint32_t op_attr_mask;

    op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_REQUEST |
                   UCP_OP_ATTR_FLAG_NO_IMM_CMPL | UCP_OP_ATTR_FIELD_USER_DATA;

    send_param.op_attr_mask = op_attr_mask;
    recv_param.op_attr_mask = op_attr_mask;
    send_param.request      = request_alloc();
    recv_param.request      = request_alloc();
    send_param.cb.send      = send_callback;
    recv_param.cb.recv      = recv_callback;
    send_param.user_data    = &completed;
    recv_param.user_data    = &completed;

    std::vector<char> send_buffer(MSG_SIZE);
    std::vector<char> recv_buffer(MSG_SIZE);

    completion_value completion(&completed, 2u);

    test_recv_send(MSG_SIZE, &send_buffer[0], &recv_buffer[0], send_param,
                   recv_param, completion);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_nbx)
