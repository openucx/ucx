/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <ucs/datastruct/queue.h>
}

#include <common/test_helpers.h>
#include <iostream>


class test_ucp_tag_xfer : public test_ucp_tag {
public:
    enum {
        VARIANT_DEFAULT,
        VARIANT_ERR_HANDLING,
        VARIANT_RNDV_PUT_ZCOPY,
        VARIANT_RNDV_AUTO,
        VARIANT_SEND_NBR,
    };

    virtual void init() {
        if (GetParam().variant == VARIANT_RNDV_PUT_ZCOPY) {
            modify_config("RNDV_SCHEME", "put_zcopy");
        } else if (GetParam().variant == VARIANT_RNDV_AUTO) {
            modify_config("RNDV_SCHEME", "auto");
        }
        test_ucp_tag::init();
    }

    std::vector<ucp_test_param>
    static enum_test_params(const ucp_params_t& ctx_params,
                            const std::string& name,
                            const std::string& test_case_name,
                            const std::string& tls)
    {
        std::vector<ucp_test_param> result;
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     VARIANT_DEFAULT, result);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name + "/err_handling_mode_peer",
                                     tls, VARIANT_ERR_HANDLING, result);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name + "/rndv_put_zcopy", tls,
                                     VARIANT_RNDV_PUT_ZCOPY, result);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name + "/rndv_auto", tls,
                                     VARIANT_RNDV_AUTO, result);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name + "/send_nbr", tls,
                                     VARIANT_SEND_NBR, result);
        return result;
    }

    virtual ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t ep_params = test_ucp_tag::get_ep_params();
        if (GetParam().variant == VARIANT_ERR_HANDLING) {
            ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
            ep_params.err_mode    = UCP_ERR_HANDLING_MODE_PEER;
        }
        return ep_params;
    }

    bool is_err_handling() const {
        return GetParam().variant == VARIANT_ERR_HANDLING;
    }

    void skip_err_handling() const {
        if (is_err_handling()) {
            UCS_TEST_SKIP_R("err_handling");
        }
    }

    void test_xfer_contig(size_t size, bool expected, bool sync, bool truncated);
    void test_xfer_generic(size_t size, bool expected, bool sync, bool truncated);
    void test_xfer_iov(size_t size, bool expected, bool sync, bool truncated);
    void test_xfer_generic_err(size_t size, bool expected, bool sync, bool truncated);

protected:
    typedef void (test_ucp_tag_xfer::* xfer_func_t)(size_t size, bool expected,
                                                    bool sync, bool truncated);

    void test_xfer(xfer_func_t func, bool expected, bool sync, bool truncated);
    void test_run_xfer(bool send_contig, bool recv_contig,
                       bool expected, bool sync, bool truncated);
    void test_xfer_prepare_bufs(uint8_t *sendbuf, uint8_t *recvbuf, size_t count,
                                bool send_contig, bool recv_contig,
                                ucp_datatype_t *send_dt,
                                ucp_datatype_t *recv_dt);
    void test_xfer_probe(bool send_contig, bool recv_contig,
                         bool expected, bool sync);

private:
    size_t do_xfer(const void *sendbuf, void *recvbuf, size_t count,
                   ucp_datatype_t send_dt, ucp_datatype_t recv_dt,
                   bool expected, bool sync, bool truncated);

    request* do_send(const void *sendbuf, size_t count, ucp_datatype_t dt, bool sync);

    static const uint64_t SENDER_TAG = 0x111337;
    static const uint64_t RECV_MASK  = 0xffff;
    static const uint64_t RECV_TAG   = 0x1337;

};

int check_buffers(const std::vector<char> &sendbuf, const std::vector<char> &recvbuf,
                  size_t recvd, size_t send_iovcnt, size_t recv_iovcnt,
                  size_t size, bool expected, bool sync, const std::string datatype)
{
    int buffers_equal = memcmp(sendbuf.data(), recvbuf.data(), recvd);
    if (buffers_equal) {
        std::cout << "\n";
        ucs::detail::message_stream ms("INFO");
        for (size_t it = 0; it < recvd; ++it) {
            if (sendbuf[it] != recvbuf[it]) {
                ms << datatype << ':'
                   << " send_iovcnt=" << std::dec << send_iovcnt
                   << " recv_iovcnt=" << recv_iovcnt << " size=" << size
                   << " expected=" << expected << " sync=" << sync
                   << " Sendbuf[" << std::dec << it << "]=0x"
                   << std::hex << (static_cast<int>(sendbuf[it]) & 0xff) << ','
                   << " Recvbuf[" << std::dec << it << "]=0x"
                   << std::hex << (static_cast<int>(recvbuf[it]) & 0xff) << std::endl;
                break;
            }
        }
    }
    return buffers_equal;
}

void test_ucp_tag_xfer::test_xfer(xfer_func_t func, bool expected, bool sync,
                                  bool truncated)
{
    if (sync) {
        skip_err_handling();
    }

    ucs::detail::message_stream ms("INFO");

    ms << "0 " << std::flush;
    (this->*func)(0, expected, sync, false);

    for (unsigned i = 1; i <= 7; ++i) {
        size_t max = (long)pow(10.0, i);

        long count = ucs_max((long)(5000.0 / sqrt(max) / ucs::test_time_multiplier()),
                             3);
        if (!expected) {
            count = ucs_min(count, 50);
        }
        ms << count << "x10^" << i << " " << std::flush;
        for (long j = 0; j < count; ++j) {
            size_t size = ucs::rand() % max + 1;
            (this->*func)(size, expected, sync, truncated);
        }
    }
}

void test_ucp_tag_xfer::test_xfer_prepare_bufs(uint8_t *sendbuf, uint8_t *recvbuf,
                                               size_t count, bool send_contig,
                                               bool recv_contig,
                                               ucp_datatype_t *send_dt,
                                               ucp_datatype_t *recv_dt)
{
    ucs_status_t status;

    if (send_contig) {
        /* the sender has a contig datatype for the data buffer */
        for (unsigned i = 0; i < count; ++i) {
             sendbuf[i] = i % 256;
        }
        *send_dt = DATATYPE;
    } else {
        /* the sender has a generic datatype */
        status = ucp_dt_create_generic(&test_dt_uint8_ops, NULL, send_dt);
        ASSERT_UCS_OK(status);
    }

    if (recv_contig) {
        /* the recv has a contig datatype for the data buffer */
        *recv_dt = DATATYPE;
    } else {
        /* the receiver has a generic datatype */
        status = ucp_dt_create_generic(&test_dt_uint8_ops, NULL, recv_dt);
        /* the recvbuf can be NULL because we only validate the received data in the
        * unpack function - we don't copy it to the recvbuf */
        ASSERT_UCS_OK(status);
    }
}

void test_ucp_tag_xfer::test_run_xfer(bool send_contig, bool recv_contig,
                                      bool expected, bool sync, bool truncated)
{
    static const size_t count = 1148544 / ucs::test_time_multiplier();
    uint8_t *sendbuf = NULL, *recvbuf = NULL;
    ucp_datatype_t send_dt, recv_dt;
    size_t recvd;

    if (sync) {
        skip_err_handling();
    }

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;

    if (send_contig) {
        /* the sender has a contig datatype for the data buffer */
        sendbuf = (uint8_t*) malloc(count * sizeof(*sendbuf));
    }
    if (recv_contig) {
        /* the recv has a contig datatype for the data buffer */
        recvbuf = (uint8_t*) malloc(count * sizeof(*recvbuf));
    }

    test_xfer_prepare_bufs(sendbuf, recvbuf, count, send_contig, recv_contig,
                           &send_dt, &recv_dt);

    /* coverity[var_deref_model] */
    recvd = do_xfer(&sendbuf[0], &recvbuf[0], count, send_dt, recv_dt, expected,
                    sync, truncated);
    if (!truncated) {
        EXPECT_EQ(count * sizeof(uint8_t), recvd);
    }

    if (send_contig) {
        free(sendbuf);
    } else {
        ucp_dt_destroy(send_dt);
    }

    if (recv_contig) {
        free(recvbuf);
    } else {
        ucp_dt_destroy(recv_dt);
    }
}

void test_ucp_tag_xfer::test_xfer_probe(bool send_contig, bool recv_contig,
                                        bool expected, bool sync)
{
    static const size_t count = 1148544 / ucs::test_time_multiplier();
    uint8_t             *sendbuf = NULL;
    uint8_t             *recvbuf = NULL;
    ucp_datatype_t      send_dt, recv_dt;
    ucp_tag_message_h   message;
    ucp_tag_recv_info_t info;
    request             *rreq, *sreq;

    if (&sender() == &receiver()) {
        /* the self transport doesn't do rndv and completes the send immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;

    sendbuf = (uint8_t*) malloc(count * sizeof(*sendbuf));
    recvbuf = (uint8_t*) malloc(count * sizeof(*recvbuf));

    test_xfer_prepare_bufs(sendbuf, recvbuf, count, send_contig, recv_contig,
                           &send_dt, &recv_dt);

    info.length = 0;
    message = ucp_tag_probe_nb(receiver().worker(), 0x1337, 0xffff, 1, &info);
    EXPECT_TRUE(message == NULL);

    sreq = send_nb(&sendbuf[0], count, send_dt, SENDER_TAG);
    EXPECT_TRUE(!UCS_PTR_IS_ERR(sreq));
    if (sreq != NULL) {
        EXPECT_FALSE(sreq->completed);
    }

    /* put RTS into the unexpected queue */
    ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(10.0);
    do {
        short_progress_loop();
        message = ucp_tag_probe_nb(receiver().worker(), RECV_TAG, RECV_MASK, 1, &info);
    } while ((ucs_get_time() < loop_end_limit) && (message == NULL));

    /* make sure that there was a match (RTS) */
    EXPECT_TRUE(message != NULL);
    EXPECT_EQ(count, info.length);
    EXPECT_EQ((ucp_tag_t)SENDER_TAG, info.sender_tag);

    /* coverity[var_deref_model] */
    rreq = (request*) ucp_tag_msg_recv_nb(receiver().worker(), &recvbuf[0],
                                          count, recv_dt, message, recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq));

    wait(rreq);
    if (sreq != NULL) {
        wait(sreq);
        request_release(sreq);
    }
    request_release(rreq);

    free(sendbuf);
    free(recvbuf);
    if (!send_contig) {
        ucp_dt_destroy(send_dt);
    }
    if (!recv_contig) {
        ucp_dt_destroy(recv_dt);
    }
}

void test_ucp_tag_xfer::test_xfer_contig(size_t size, bool expected, bool sync,
                                         bool truncated)
{
    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);
    size_t recvd = do_xfer(&sendbuf[0], &recvbuf[0], size, DATATYPE, DATATYPE,
                           expected, sync, truncated);
    if (!truncated) {
        ASSERT_EQ(sendbuf.size(), recvd);
    }
    EXPECT_TRUE(!check_buffers(sendbuf, recvbuf, recvd, 1, 1,
                               size, expected, sync, "contig"));
}

void test_ucp_tag_xfer::test_xfer_generic(size_t size, bool expected, bool sync,
                                          bool truncated)
{
    size_t count = size / sizeof(uint32_t);
    ucp_datatype_t dt;
    ucs_status_t status;
    size_t recvd;

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;

    /* if count is zero, truncation has no effect */
    if ((truncated) && (!count)) {
        truncated = false;
    }

    status = ucp_dt_create_generic(&test_dt_uint32_ops, this, &dt);
    ASSERT_UCS_OK(status);

    recvd = do_xfer(NULL, NULL, count, dt, dt, expected, sync, truncated);
    if (!truncated) {
        EXPECT_EQ(count * sizeof(uint32_t), recvd);
    }
    EXPECT_EQ(2, dt_gen_start_count);
    EXPECT_EQ(2, dt_gen_finish_count);

    ucp_dt_destroy(dt);
}

void test_ucp_tag_xfer::test_xfer_iov(size_t size, bool expected, bool sync,
                                      bool truncated)
{
    const size_t iovcnt = 20;
    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    ucp::data_type_desc_t send_dt_desc(DATATYPE_IOV, sendbuf.data(),
                                       sendbuf.size(), iovcnt);
    ucp::data_type_desc_t recv_dt_desc(DATATYPE_IOV, recvbuf.data(),
                                       recvbuf.size(), iovcnt);

    size_t recvd = do_xfer(send_dt_desc.buf(), recv_dt_desc.buf(), iovcnt,
                           DATATYPE_IOV, DATATYPE_IOV, expected, sync,
                           truncated);
    if (!truncated) {
        ASSERT_EQ(sendbuf.size(), recvd);
    }
    EXPECT_TRUE(!check_buffers(sendbuf, recvbuf, recvd, send_dt_desc.count(),
                               recv_dt_desc.count(), size, expected, sync,
                               "IOV"));
}

void test_ucp_tag_xfer::test_xfer_generic_err(size_t size, bool expected,
                                              bool sync, bool truncated)
{
    size_t count = size / sizeof(uint32_t);
    ucp_datatype_t dt;
    ucs_status_t status;
    request *rreq, *sreq;

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;

    status = ucp_dt_create_generic(&test_dt_uint32_err_ops, this, &dt);
    ASSERT_UCS_OK(status);

    if (expected) {
        rreq = recv_nb(NULL, count, dt, RECV_TAG, RECV_MASK);
        sreq = do_send(NULL, count, dt, sync);
    } else {
        sreq = do_send(NULL, count, dt, sync);
        short_progress_loop();
        if (sync) {
            EXPECT_FALSE(sreq->completed);
        }
        rreq = recv_nb(NULL, count, dt, RECV_TAG, RECV_MASK);
    }

    /* progress both sender and receiver */
    wait(rreq);
    if (sreq != NULL) {
        wait(sreq);
        request_release(sreq);
    }

    /* the generic unpack function is expected to fail */
    EXPECT_EQ(UCS_ERR_NO_MEMORY, rreq->status);
    request_release(rreq);
    EXPECT_EQ(2, dt_gen_start_count);
    EXPECT_EQ(2, dt_gen_finish_count);
    ucp_dt_destroy(dt);
}

test_ucp_tag_xfer::request*
test_ucp_tag_xfer::do_send(const void *sendbuf, size_t count, ucp_datatype_t dt,
                           bool sync)
{
    if (sync) {
        return send_sync_nb(sendbuf, count, dt, SENDER_TAG);
    } else {
        if (GetParam().variant == VARIANT_SEND_NBR) {
            return send_nbr(sendbuf, count, dt, SENDER_TAG);
        }
        return send_nb(sendbuf, count, dt, SENDER_TAG);
    }
}

size_t test_ucp_tag_xfer::do_xfer(const void *sendbuf, void *recvbuf,
                                  size_t count, ucp_datatype_t send_dt,
                                  ucp_datatype_t recv_dt, bool expected,
                                  bool sync, bool truncated)
{
    request *rreq, *sreq;
    size_t recvd = 0;
    size_t recv_count = count;

    if (truncated) {
        recv_count /= 2;
    }

    if (expected) {
        rreq = recv_nb(recvbuf, recv_count, recv_dt, RECV_TAG, RECV_MASK);
        sreq = do_send(sendbuf, count, send_dt, sync);
    } else {
        sreq = do_send(sendbuf, count, send_dt, sync);

        wait_for_unexpected_msg(receiver().worker(), 10.0);

        if (sync) {
            EXPECT_FALSE(sreq->completed);
        }
        rreq = recv_nb(recvbuf, recv_count, recv_dt, RECV_TAG, RECV_MASK);
    }

    /* progress both sender and receiver */
    wait(rreq);
    if (sreq != NULL) {
        wait(sreq);
        request_release(sreq);
    }

    recvd = rreq->info.length;
    if (!truncated) {
        ASSERT_UCS_OK(rreq->status);
        EXPECT_EQ((ucp_tag_t)SENDER_TAG, rreq->info.sender_tag);
    } else {
        EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, rreq->status);
    }

    request_release(rreq);
    return recvd;
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_truncated, "RC_TM_ENABLE?=n") {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_exp_truncated) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_exp_truncated) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_exp_sync) {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_sync) {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_exp_sync) {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_exp_sync) {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, false, true, false);
}

/* send_contig_recv_contig */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, true, true, false, false);
}

/* send_generic_recv_generic */

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(false, false, true, false, false);
}

/* send_contig_recv_generic */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_sync, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync, "RNDV_THRESH=1248576") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(true, false, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, false, false, false);
}

/* send_generic_recv_contig */

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(false, true, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp_sync, "RNDV_THRESH=1248576") {
    test_run_xfer(false, true, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp_sync, "RNDV_THRESH=1248576") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(false, true, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp, "RNDV_THRESH=1248576",
        "ZCOPY_THRESH=1248576") {
    test_run_xfer(false, true, false, false, false);
}

/* rndv send_config_recv_config am_rndv with bcopy on the sender side
 * (zcopy is tested in the match tests) */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp_rndv, "RNDV_THRESH=1000",
                                                                "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, true, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp_rndv_truncated, "RNDV_THRESH=1000",
                                                                          "ZCOPY_THRESH=1248576",
                                                                          "RC_TM_ENABLE?=n") {
    test_run_xfer(true, true, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp_sync_rndv, "RNDV_THRESH=1000",
                                                                     "ZCOPY_THRESH=1248576") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(true, true, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp_sync_rndv_truncated,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(true, true, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_unexp_rndv, "RNDV_THRESH=1000",
                                                                  "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, true, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_unexp_rndv_truncated, "RNDV_THRESH=1000",
                                                                            "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, true, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_unexp_sync_rndv, "RNDV_THRESH=1000",
                                                                        "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, true, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_unexp_sync_rndv_truncated,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, true, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp_rndv_probe, "RNDV_THRESH=1000",
                                                                      "ZCOPY_THRESH=1248576") {
    test_xfer_probe(true, true, true, false);
}

/* rndv send_generic_recv_generic am_rndv with bcopy on the sender side */

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp_rndv, "RNDV_THRESH=1000") {
    test_run_xfer(false, false, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp_rndv_truncated, "RNDV_THRESH=1000") {
    test_run_xfer(false, false, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp_sync_rndv, "RNDV_THRESH=1000") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(false, false, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp_sync_rndv_truncated,
           "RNDV_THRESH=1000") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(false, false, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_unexp_rndv, "RNDV_THRESH=1000") {
    test_run_xfer(false, false, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_unexp_rndv_truncated, "RNDV_THRESH=1000") {
    test_run_xfer(false, false, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_unexp_sync_rndv, "RNDV_THRESH=1000") {
    test_run_xfer(false, false, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_unexp_sync_rndv_truncated,
           "RNDV_THRESH=1000") {
    test_run_xfer(false, false, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp_rndv_probe, "RNDV_THRESH=1000") {
    test_xfer_probe(false, false, true, false);
}

/* rndv send_generic_recv_contig am_rndv with bcopy on the sender side */

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp_rndv, "RNDV_THRESH=1000") {
    test_run_xfer(false, true, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp_rndv_truncated, "RNDV_THRESH=1000") {
    test_run_xfer(false, true, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp_sync_rndv, "RNDV_THRESH=1000") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(false, true, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp_sync_rndv_truncated,
           "RNDV_THRESH=1000") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(false, true, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp_rndv, "RNDV_THRESH=1000") {
    test_run_xfer(false, true, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp_rndv_truncated, "RNDV_THRESH=1000") {
    test_run_xfer(false, true, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp_sync_rndv, "RNDV_THRESH=1000") {
    test_run_xfer(false, true, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp_sync_rndv_truncated,
           "RNDV_THRESH=1000") {
    test_run_xfer(false, true, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp_rndv_probe, "RNDV_THRESH=1000") {
    test_xfer_probe(false, true, true, false);
}

/* rndv send_contig_recv_generic am_rndv with bcopy on the sender side */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv, "RNDV_THRESH=1000",
                                                                 "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, false, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv_truncated,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, false, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync_rndv,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(true, false, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync_rndv_truncated,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(true, false, true, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_rndv,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, false, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_rndv_truncated,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, false, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_sync_rndv,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, false, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_sync_rndv_truncated,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    test_run_xfer(true, false, false, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv_probe, "RNDV_THRESH=1000",
                                                                       "ZCOPY_THRESH=1248576") {
    test_xfer_probe(true, false, true, false);
}

/* rndv send_contig_recv_generic am_rndv with zcopy on the sender side */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv_zcopy, "RNDV_THRESH=1000",
                                                                       "ZCOPY_THRESH=1000") {
    test_run_xfer(true, false, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv_truncated_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    test_run_xfer(true, false, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync_rndv_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(true, false, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync_rndv_truncated_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_run_xfer(true, false, true, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_rndv_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    test_run_xfer(true, false, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_rndv_truncated_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    test_run_xfer(true, false, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_sync_rndv_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    test_run_xfer(true, false, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_sync_rndv_truncated_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    test_run_xfer(true, false, false, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv_probe_zcopy, "RNDV_THRESH=1000",
                                                                             "ZCOPY_THRESH=1000") {
    test_xfer_probe(true, false, true, false);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_xfer)


#if ENABLE_STATS

class test_ucp_tag_stats : public test_ucp_tag_xfer {
public:
    void init() {
        stats_activate();
        test_ucp_tag_xfer::init();
    }

    void cleanup() {
        test_ucp_tag_xfer::cleanup();
        stats_restore();
    }

    std::vector<ucp_test_param>
    static enum_test_params(const ucp_params_t& ctx_params,
                            const std::string& name,
                            const std::string& test_case_name,
                            const std::string& tls) {

        return ucp_test::enum_test_params(ctx_params, name,
                                          test_case_name, tls);
    }

    ucs_stats_node_t* ep_stats(entity &e) {
        return e.ep()->stats;
    }

    ucs_stats_node_t* worker_stats(entity &e) {
        return e.worker()->stats;
    }

    void validate_counters(uint64_t tx_cntr, uint64_t rx_cntr) {
        uint64_t cnt;
        cnt = UCS_STATS_GET_COUNTER(ep_stats(sender()), tx_cntr);
        EXPECT_EQ(1ul, cnt);
        cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()), rx_cntr);
        EXPECT_EQ(1ul, cnt);
    }

};


UCS_TEST_P(test_ucp_tag_stats, eager_expected, "RNDV_THRESH=1248576",
                                               "RC_TM_ENABLE?=n") {
    test_run_xfer(true, true, true, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER,
                      UCP_WORKER_STAT_TAG_RX_EAGER_MSG);

    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_EQ(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, eager_unexpected, "RNDV_THRESH=1248576",
                                                 "RC_TM_ENABLE?=n") {
    test_run_xfer(true, true, false, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER,
                      UCP_WORKER_STAT_TAG_RX_EAGER_MSG);
    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_GT(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, sync_expected, "RNDV_THRESH=1248576",
                                              "RC_TM_ENABLE?=n") {
    skip_loopback();
    test_run_xfer(true, true, true, true, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER_SYNC,
                      UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG);

    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                 UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_EQ(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, sync_unexpected, "RNDV_THRESH=1248576",
                                                "RC_TM_ENABLE?=n") {
    skip_loopback();
    test_run_xfer(true, true, false, true, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER_SYNC,
                      UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG);
    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_GT(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, rndv_expected, "RNDV_THRESH=1000",
                                              "RC_TM_ENABLE?=n") {
    test_run_xfer(true, true, true, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_RNDV,
                      UCP_WORKER_STAT_TAG_RX_RNDV_EXP);
}

UCS_TEST_P(test_ucp_tag_stats, rndv_unexpected, "RNDV_THRESH=1000",
                                                "RC_TM_ENABLE?=n") {
    test_run_xfer(true, true, false, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_RNDV,
                      UCP_WORKER_STAT_TAG_RX_RNDV_UNEXP);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_stats)

#endif
