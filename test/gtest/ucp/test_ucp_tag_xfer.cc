/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>
#include <iostream>


class test_ucp_tag_xfer : public test_ucp_tag {
public:
    using test_ucp_tag::get_ctx_params;

    void test_xfer_contig(size_t size, bool expected);
    void test_xfer_generic(size_t size, bool expected);

protected:
    typedef void (test_ucp_tag_xfer::* xfer_func_t)(size_t size, bool expected);

    void test_xfer(xfer_func_t func, bool expected);

private:
    size_t do_xfer(const void *sendbuf, void *recvbuf, ucp_datatype_t dt,
                   size_t count, bool expected);
    request *recv_nb(void *buffer, size_t count, ucp_datatype_t dt,
                     ucp_tag_t tag, ucp_tag_t tag_mask);
};

void test_ucp_tag_xfer::test_xfer(xfer_func_t func, bool expected)
{
    ucs::detail::message_stream ms("INFO");

    ms << "0 " << std::flush;
    (this->*func)(0, expected);

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
            (this->*func)(size, expected);
        }
    }
}

void test_ucp_tag_xfer::test_xfer_contig(size_t size, bool expected)
{
    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());
    size_t recvd = do_xfer(&sendbuf[0], &recvbuf[0], size, DATATYPE, expected);

    ASSERT_EQ(sendbuf.size(), recvd);
    EXPECT_TRUE(!memcmp(&sendbuf[0], &recvbuf[0], recvd));
}

void test_ucp_tag_xfer::test_xfer_generic(size_t size, bool expected)
{
    size_t count = size / sizeof(uint32_t);
    ucp_datatype_t dt;
    ucs_status_t status;
    size_t recvd;

    dt_gen_start_count  = 0;
    dt_gen_finish_count = 0;

    status = ucp_dt_create_generic(&test_dt_ops, this, &dt);
    ASSERT_UCS_OK(status);

    recvd = do_xfer(NULL, NULL, count, dt, expected);
    EXPECT_EQ(count * sizeof(uint32_t), recvd);

    EXPECT_EQ(2, dt_gen_start_count);
    EXPECT_EQ(2, dt_gen_finish_count);

    ucp_dt_destroy(dt);
}

test_ucp_tag_xfer::request*
test_ucp_tag_xfer::recv_nb(void *buffer, size_t count, ucp_datatype_t dt,
                           ucp_tag_t tag, ucp_tag_t tag_mask)
{
    request *req = (request*)ucp_tag_recv_nb(receiver->worker(), buffer, count,
                                             dt, tag, tag_mask, recv_callback);
    if (UCS_PTR_IS_ERR(req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    } else if (req == NULL) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
    }
    return req;
}

size_t test_ucp_tag_xfer::do_xfer(const void *sendbuf, void *recvbuf, size_t count,
                                  ucp_datatype_t dt, bool expected)
{
    request *req;
    size_t recvd;

    if (expected) {
        req = recv_nb(recvbuf, count, dt, 0x1337, 0xffff);
        send_b(sendbuf, count, dt, 0x111337);
    } else {
        send_b(sendbuf, count, dt, 0x111337);
        short_progress_loop();
        req = recv_nb(recvbuf, count, dt, 0x1337, 0xffff);
    }

    wait(req);

    ASSERT_UCS_OK(req->status);
    EXPECT_EQ((ucp_tag_t)0x111337, req->info.sender_tag);
    recvd = req->info.length;
    request_release(req);
    return recvd;
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_unexp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_xfer)
