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

    void test_xfer_contig(size_t size, bool expected, bool sync);
    void test_xfer_generic(size_t size, bool expected, bool sync);
    void test_xfer_iov(size_t size, bool expected, bool sync);

protected:
    typedef void (test_ucp_tag_xfer::* xfer_func_t)(size_t size, bool expected,
                                                    bool sync);

    void test_xfer(xfer_func_t func, bool expected, bool sync);
    void test_run_xfer(bool send_contig, bool recv_contig,
                       bool expected, bool sync);

private:
    size_t do_xfer(const void *sendbuf, void *recvbuf, size_t count,
                   ucp_datatype_t send_dt, ucp_datatype_t recv_dt,
                   bool expected, bool sync);

    request* do_send(const void *sendbuf, size_t count, ucp_datatype_t dt, bool sync);

    static const uint64_t SENDER_TAG = 0x111337;
    static const uint64_t RECV_MASK  = 0xffff;
    static const uint64_t RECV_TAG   = 0x1337;

};

void test_ucp_tag_xfer::test_xfer(xfer_func_t func, bool expected, bool sync)
{
    ucs::detail::message_stream ms("INFO");

    ms << "0 " << std::flush;
    (this->*func)(0, expected, sync);

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
            (this->*func)(size, expected, sync);
        }
    }
}

void test_ucp_tag_xfer::test_run_xfer(bool send_contig, bool recv_contig,
                                      bool expected, bool sync)
{
    static const size_t count = 1148544 / ucs::test_time_multiplier();
    uint8_t *sendbuf, *recvbuf;
    ucp_datatype_t send_dt, recv_dt;
    ucs_status_t status;
    size_t recvd;

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;

    if (send_contig) {
        /* the sender has a contig datatype for the data buffer */
        sendbuf = (uint8_t*)malloc(count * sizeof(*sendbuf));
        for (unsigned i = 0; i < count; ++i) {
             sendbuf[i] = i % 256;
        }
        send_dt = DATATYPE;
    } else {
        /* the sender has a generic datatype */
        status = ucp_dt_create_generic(&test_dt_uint8_ops, NULL, &send_dt);
        ASSERT_UCS_OK(status);
        sendbuf = NULL;
    }

    if (recv_contig) {
        /* the recv has a contig datatype for the data buffer */
        recvbuf = (uint8_t*)malloc(count * sizeof(*recvbuf));
        recv_dt = DATATYPE;
    } else {
        /* the receiver has a generic datatype */
        status = ucp_dt_create_generic(&test_dt_uint8_ops, NULL, &recv_dt);
        /* the recvbuf can be NULL because we only validate the received data in the
         * unpack function - we don't copy it to the recvbuf */
        ASSERT_UCS_OK(status);
        recvbuf = NULL;
    }

    /* coverity[var_deref_model] */
    recvd = do_xfer(&sendbuf[0], &recvbuf[0], count, send_dt, recv_dt, expected, sync);
    EXPECT_EQ(count * sizeof(uint8_t), recvd);

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

void test_ucp_tag_xfer::test_xfer_contig(size_t size, bool expected, bool sync)
{
    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);
    size_t recvd = do_xfer(&sendbuf[0], &recvbuf[0], size, DATATYPE, DATATYPE,
                           expected, sync);
    ASSERT_EQ(sendbuf.size(), recvd);
    EXPECT_TRUE(!memcmp(&sendbuf[0], &recvbuf[0], recvd));
}

void test_ucp_tag_xfer::test_xfer_generic(size_t size, bool expected, bool sync)
{
    size_t count = size / sizeof(uint32_t);
    ucp_datatype_t dt;
    ucs_status_t status;
    size_t recvd;

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;

    status = ucp_dt_create_generic(&test_dt_uint32_ops, this, &dt);
    ASSERT_UCS_OK(status);

    recvd = do_xfer(NULL, NULL, count, dt, dt, expected, sync);
    EXPECT_EQ(count * sizeof(uint32_t), recvd);

    EXPECT_EQ(2, dt_gen_start_count);
    EXPECT_EQ(2, dt_gen_finish_count);

    ucp_dt_destroy(dt);
}

void test_ucp_tag_xfer::test_xfer_iov(size_t size, bool expected, bool sync)
{
    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    UCS_TEST_GET_BUFFER_DT_IOV(iov, iovcnt, sendbuf.data(), sendbuf.size(), 20);

    size_t recvd = do_xfer(&iov, &recvbuf[0], iovcnt, DATATYPE_IOV, DATATYPE_IOV,
                           expected, sync);

    ASSERT_EQ(sendbuf.size(), recvd);
    EXPECT_TRUE(!memcmp(sendbuf.data(), recvbuf.data(), recvd));
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
                                  ucp_datatype_t recv_dt, bool expected, bool sync)
{
    request *rreq, *sreq;
    size_t recvd;
    size_t recv_count = count;

    if (UCP_DATATYPE_IOV == (recv_dt & UCP_DATATYPE_CLASS_MASK)) {
        recv_dt = DATATYPE;
        recv_count = ucp_dt_iov_length((const ucp_dt_iov_t *)sendbuf, count);
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

    wait(rreq);
    if (sreq != NULL) {
        wait(sreq);
        request_release(sreq);
    }

    ASSERT_UCS_OK(rreq->status);
    EXPECT_EQ((ucp_tag_t)SENDER_TAG, rreq->info.sender_tag);
    recvd = rreq->info.length;
    request_release(rreq);
    return recvd;
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_sync) {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_exp_sync) {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_exp_sync) {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, iov_unexp_sync) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_iov, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_contig_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_generic_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(false, false, true, false);
}

/* send_contig_recv_contig */

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp_sync, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_exp_sync, "RNDV_THRESH=1248576") {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_run_xfer(true, false, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_contig_recv_generic_unexp, "RNDV_THRESH=1248576") {
    test_run_xfer(true, false, false, false);
}

/* send_generic_recv_contig */

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp, "RNDV_THRESH=1248576") {
    test_run_xfer(false, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp_sync, "RNDV_THRESH=1248576") {
    test_run_xfer(false, true, false, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_exp_sync, "RNDV_THRESH=1248576") {
    if (&sender() == &receiver()) { /* because ucp_tag_send_req return status
                                       (instead request) if send operation
                                       completed immediately */
        UCS_TEST_SKIP_R("loop-back unsupported");
    }
    test_run_xfer(false, true, true, true);
}

UCS_TEST_P(test_ucp_tag_xfer, send_generic_recv_contig_unexp, "RNDV_THRESH=1248576") {
    test_run_xfer(false, true, false, false);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_xfer)
