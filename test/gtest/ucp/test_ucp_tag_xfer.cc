/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <ucp/dt/dt.h>

#include <common/test_helpers.h>
#include <iostream>


class test_ucp_tag_xfer : public test_ucp_tag {
public:
    using test_ucp_tag::get_ctx_params;

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

void test_ucp_tag_xfer::test_xfer(xfer_func_t func, bool expected, bool sync,
                                  bool truncated)
{
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
            size_t size = rand() % max + 1;
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
        sendbuf = NULL;
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
        recvbuf = NULL;
    }
}

void test_ucp_tag_xfer::test_run_xfer(bool send_contig, bool recv_contig,
                                      bool expected, bool sync, bool truncated)
{
    static const size_t count = 1148544 / ucs::test_time_multiplier();
    uint8_t *sendbuf = NULL, *recvbuf = NULL;
    ucp_datatype_t send_dt, recv_dt;
    size_t recvd;

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
    uint8_t *sendbuf = NULL, *recvbuf = NULL;
    ucp_datatype_t send_dt, recv_dt;
    ucp_tag_message_h message;
    ucp_tag_recv_info info;
    request *rreq, *sreq;

    if (&sender() == &receiver()) {
        /* the self transport doesn't do rndv and completes the send immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
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

    info.length = 0;
    message = ucp_tag_probe_nb(receiver().worker(), 0x1337, 0xffff, 1, &info);
    EXPECT_TRUE(message == NULL);

    sreq = send_nb(&sendbuf[0], count, send_dt, SENDER_TAG);
    EXPECT_TRUE(!UCS_PTR_IS_ERR(sreq));
    if (sreq != NULL) {
        EXPECT_FALSE(sreq->completed);
    }

    /* put RTS into the unexpected queue */
    wait_for_flag(&info.length);

    message = ucp_tag_probe_nb(receiver().worker(), RECV_TAG, RECV_MASK, 1, &info);
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
    EXPECT_TRUE(!memcmp(&sendbuf[0], &recvbuf[0], recvd));
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

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    UCS_TEST_GET_BUFFER_DT_IOV(send_iov, send_iovcnt, sendbuf.data(), sendbuf.size(), iovcnt);
    UCS_TEST_GET_BUFFER_DT_IOV(recv_iov, recv_iovcnt, recvbuf.data(), recvbuf.size(), iovcnt);

    size_t recvd = do_xfer(&send_iov, &recv_iov, iovcnt, DATATYPE_IOV, DATATYPE_IOV,
                           expected, sync, truncated);
    if (!truncated) {
        ASSERT_EQ(sendbuf.size(), recvd);
    }
    EXPECT_TRUE(!memcmp(sendbuf.data(), recvbuf.data(), recvd));
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
        short_progress_loop();
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

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_truncated) {
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
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_sync) {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_exp_sync) {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_exp_sync) {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, true, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(false, false, true, false, false);
}

/* send_contig_recv_contig */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_sync, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync, "RNDV_THRESH=1248576") {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
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
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_run_xfer(false, true, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp, "RNDV_THRESH=1248576") {
    test_run_xfer(false, true, false, false, false);
}

/* rndv bcopy */

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
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_run_xfer(true, false, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync_rndv_truncated,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1248576") {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
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

/* rndv bcopy probe */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv_probe, "RNDV_THRESH=1000",
                                                                       "ZCOPY_THRESH=1248576") {
    test_xfer_probe(true, false, true, false);
}

/* rndv zcopy */

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
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_run_xfer(true, false, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync_rndv_truncated_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_run_xfer(true, false, true, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_rndv_zcopy, "RNDV_THRESH=1000",
                                                                         "ZCOPY_THRESH=1000") {
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

/* rndv zcopy probe */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_rndv_probe_zcopy,
           "RNDV_THRESH=1000", "ZCOPY_THRESH=1000") {
    test_xfer_probe(true, false, true, false);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_xfer)
