/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucs/debug/debug.h>
}

using namespace ucs; /* For vector<char> serialization */

class test_ucp_tag : public ucp_test {
protected:
    struct request {
        bool                completed;
        ucs_status_t        status;
        ucp_tag_recv_info_t info;
    };

    struct dt_gen_state {
        size_t              count;
        int                 started;
        uint32_t            magic;
    };

    virtual void init() {
        ucp_test::init();
        sender   = create_entity();
        receiver = create_entity();
        sender->connect(receiver);

        dt_gen_start_count  = 0;
        dt_gen_finish_count = 0;
    }

    virtual void cleanup() {
        sender->flush();
        receiver->flush();
        sender->disconnect();
        ucp_test::cleanup();
    }

    virtual void get_params(ucp_params_t& params) const {
        ucp_test::get_params(params);
        params.features     = UCP_FEATURE_TAG;
        params.request_size = sizeof(request);
        params.request_init = request_init;
    }

    static void request_init(void *request) {
        struct request *req = (struct request *)request;
        req->completed       = false;
        req->info.length     = 0;
        req->info.sender_tag = 0;
    }

    static void request_release(struct request *req) {
        req->completed = false;
        ucp_request_release(req);
    }

    static void send_callback(void *request, ucs_status_t status)
    {
        struct request *req = (struct request *)request;
        ucs_assert(req->completed == false);
        req->status    = status;
        req->completed = true;
    }

    static void recv_callback(void *request, ucs_status_t status,
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

    void wait(request *req)
    {
        while (!req->completed) {
            progress();
        }
    }

    void send_b(const void *buffer, size_t count, ucp_datatype_t datatype,
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

    ucs_status_t recv_b(void *buffer, size_t count, ucp_datatype_t datatype,
                        ucp_tag_t tag, ucp_tag_t tag_mask, ucp_tag_recv_info_t *info)
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

    static void* dt_start(size_t count)
    {
        dt_gen_state *dt_state = new dt_gen_state;
        dt_state->count   = count;
        dt_state->started = 1;
        dt_state->magic   = MAGIC;
        dt_gen_start_count++;
        return dt_state;
    }

    static void* dt_start_pack(void *context, const void *buffer, size_t count)
    {
        return dt_start(count);
    }

    static void* dt_start_unpack(void *context, void *buffer, size_t count)
    {
        return dt_start(count);
    }

    static size_t dt_packed_size(void *state)
    {
        dt_gen_state *dt_state = (dt_gen_state*)state;
        return dt_state->count * sizeof(uint32_t);
    }

    static size_t dt_pack(void *state, size_t offset, void *dest, size_t max_length)
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

    static ucs_status_t dt_unpack(void *state, size_t offset, const void *src,
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

    static void dt_finish(void *state)
    {
        dt_gen_state *dt_state = (dt_gen_state*)state;
        --dt_state->started;
        EXPECT_EQ(0, dt_state->started);
        dt_gen_finish_count++;
        delete dt_state;
    }

    static const uint32_t MAGIC = 0xd7d7d7d7U;
    static const ucp_datatype_t DATATYPE;
    static ucp_generic_dt_ops test_dt_ops;
    static int dt_gen_start_count;
    static int dt_gen_finish_count;

public:
    int    count;
    entity *sender, *receiver;
};

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


UCS_TEST_F(test_ucp_tag, send_recv_exp) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    /* No progress - goes to expected */

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_F(test_ucp_tag, send_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_F(test_ucp_tag, send_recv_exp_medium) {
    static const size_t size = 50000;
    ucs_status_t status;
    ucp_tag_recv_info_t info;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
}

UCS_TEST_F(test_ucp_tag, send2_nb_recv_exp_medium) {
    static const size_t size = 50000;
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    /* 1st send */

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    /* 2nd send */

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    request *my_send_req;
    my_send_req = (request*)ucp_tag_send_nb(sender->ep(), &sendbuf[0],
                                            sendbuf.size(), DATATYPE, 0x111337,
                                            send_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_release(my_send_req);
    }
}

UCS_TEST_F(test_ucp_tag, send2_nb_recv_medium_wildcard) {
    static const size_t size = 3000000;

    entity *sender1 = create_entity();
    sender1->connect(receiver);

    entity *sender2 = create_entity();
    sender2->connect(receiver);

    for (int is_exp = 0; is_exp <= 1; ++is_exp) {

        UCS_TEST_MESSAGE << "Testing " << (is_exp ? "" : "un") << "expected mode, size " << size;

        std::vector<char> sendbuf1(size, 0);
        std::vector<char> sendbuf2(size, 0);
        std::vector<char> recvbuf1(size, 0);
        std::vector<char> recvbuf2(size, 0);

        ucs::fill_random(sendbuf1.begin(), sendbuf1.end());
        ucs::fill_random(sendbuf2.begin(), sendbuf2.end());

        /* Two sends with different tags */

        request *sreq1, *sreq2;
        sreq1 = (request*)ucp_tag_send_nb(sender1->ep(), &sendbuf1[0], sendbuf1.size(),
                                          DATATYPE, 1, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(sreq1));

        sreq2 = (request*)ucp_tag_send_nb(sender2->ep(), &sendbuf2[0], sendbuf2.size(),
                                          DATATYPE, 2, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(sreq2));


        /* In unexpected mode, we progress all to put the messages on the
         *  unexpected queue
         */
        if (!is_exp) {
            short_progress_loop();
        }

        /* Two receives with any tag */

        request *rreq1, *rreq2;

        rreq1 = (request*)ucp_tag_recv_nb(receiver->worker(), &recvbuf1[0],
                                          recvbuf1.size(), DATATYPE, 0, 0,
                                          recv_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq1));

        rreq2 = (request*)ucp_tag_recv_nb(receiver->worker(), &recvbuf2[0],
                                          recvbuf2.size(), DATATYPE, 0, 0,
                                          recv_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq2));


        /* Wait for receives */
        wait(rreq1);
        wait(rreq2);

        /* Release sends */
        if (sreq1 != NULL) {
            EXPECT_TRUE(sreq1->completed);
            request_release(sreq1);
        }
        if (sreq2 != NULL) {
            EXPECT_TRUE(sreq2->completed);
            request_release(sreq2);
        }

        /* Receives should be completed with correct length */
        ASSERT_TRUE(rreq1->completed);
        ASSERT_TRUE(rreq2->completed);

        EXPECT_EQ(size, rreq1->info.length);
        EXPECT_EQ(size, rreq2->info.length);

        /* The order may be any, but the messages have to be received correctly */
        if (rreq1->info.sender_tag == 1u) {
            ASSERT_EQ(2u, rreq2->info.sender_tag);
            EXPECT_EQ(sendbuf1, recvbuf1);
            EXPECT_EQ(sendbuf2, recvbuf2);
        } else {
            ASSERT_EQ(2u, rreq1->info.sender_tag);
            ASSERT_EQ(1u, rreq2->info.sender_tag);
            EXPECT_EQ(sendbuf2, recvbuf1);
            EXPECT_EQ(sendbuf1, recvbuf2);
        }

        request_release(rreq1);
        request_release(rreq2);
    }

    sender1->flush();
    sender1->disconnect();

    sender2->flush();
    sender2->disconnect();
}

UCS_TEST_F(test_ucp_tag, send_recv_nb_partial_exp_medium) {
    static const size_t size = 50000;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    usleep(1000);
    progress();

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_recv_nb(receiver->worker(), &recvbuf[0],
                                            recvbuf.size(), DATATYPE, 0x1337,
                                            0xffff, recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    request_release(my_recv_req);
}

UCS_TEST_F(test_ucp_tag, send_recv_unexp_medium) {
    static const size_t size = 50000;
    ucs_status_t status;
    ucp_tag_recv_info_t info;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
}

UCS_TEST_F(test_ucp_tag, send_recv_exp_gentype) {
    size_t counts[3];
    counts[0] = 0;
    counts[1] = 100;
    counts[2] = 10000000 / ucs::test_time_multiplier();

    for (unsigned i = 0; i < 3; ++i) {
        size_t count = counts[i];
        ucp_datatype_t dt;
        ucp_tag_recv_info_t info;
        ucs_status_t status;

        dt_gen_start_count  = 0;
        dt_gen_finish_count = 0;

        status = ucp_dt_create_generic(&test_dt_ops, this, &dt);
        ASSERT_UCS_OK(status);

        send_b(NULL, count, dt, 0x111337);

        EXPECT_EQ(1, dt_gen_start_count);
        EXPECT_EQ(1, dt_gen_finish_count);

        status = recv_b(NULL, count, dt, 0x1337, 0xffff, &info);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(count * sizeof(uint32_t), info.length);
        EXPECT_EQ((ucp_tag_t)0x111337,      info.sender_tag);

        EXPECT_EQ(2, dt_gen_start_count);
        EXPECT_EQ(2, dt_gen_finish_count);

        ucp_dt_destroy(dt);
    }
}

UCS_TEST_F(test_ucp_tag, send_nb_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    request *my_send_req;
    my_send_req = (request*)ucp_tag_send_nb(sender->ep(), &send_data,
                                            sizeof(send_data), DATATYPE, 0x111337,
                                            send_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    ucp_worker_progress(receiver->worker());

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_release(my_send_req);
    }
}

UCS_TEST_F(test_ucp_tag, send_recv_truncated) {
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    status = recv_b(NULL, 0, DATATYPE, 0x1337, 0xffff, &info);
    EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, status);
}

UCS_TEST_F(test_ucp_tag, send_recv_nb_exp) {

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_recv_nb(receiver->worker(), &recv_data,
                                            sizeof(recv_data), DATATYPE, 0x1337,
                                            0xffff, recv_callback);

    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    ASSERT_TRUE(my_recv_req != NULL); /* Couldn't be completed because didn't send yet */

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    wait(my_recv_req);

    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sizeof(send_data),   my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
    request_release(my_recv_req);
}

UCS_TEST_F(test_ucp_tag, send_nb_multiple_recv_unexp) {
    const unsigned num_requests = 1000;
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    std::vector<request*> send_reqs(num_requests);

    for (unsigned i = 0; i < num_requests; ++i) {
        send_reqs[i] = (request*)ucp_tag_send_nb(sender->ep(), &send_data,
                                                 sizeof(send_data), DATATYPE,
                                                 0x111337, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(send_reqs[i]));
    }

    ucp_worker_progress(receiver->worker());

    for (unsigned i = 0; i < num_requests; ++i) {
        status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff,
                        &info);
        ASSERT_UCS_OK(status);
        ASSERT_EQ(num_requests, send_reqs.size());

        EXPECT_EQ(sizeof(send_data),   info.length);
        EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
        EXPECT_EQ(send_data, recv_data);
    }

    for (unsigned i = 0; i < num_requests; ++i) {
        if (send_reqs[i] != NULL) {
            EXPECT_TRUE(send_reqs[i]->completed);
            EXPECT_EQ(UCS_OK, send_reqs[i]->status);
            request_release(send_reqs[i]);
        }
    }
}

UCS_TEST_F(test_ucp_tag, send_probe) {

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;
    ucp_tag_recv_info info;
    ucp_tag_message_h message;

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 0, &info);
    EXPECT_TRUE(message == NULL);

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    do {
        message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 0, &info);
    } while (message == NULL);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_recv_nb(receiver->worker(), &recv_data,
                                            sizeof(recv_data), DATATYPE, 0x1337,
                                            0xffff, recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sizeof(send_data),   my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
    request_release(my_recv_req);
}

UCS_TEST_F(test_ucp_tag, send_medium_msg_probe) {
    static const size_t size = 50000;
    ucp_tag_recv_info info;
    ucp_tag_message_h message;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 1, &info);
    EXPECT_TRUE(message == NULL);

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    short_progress_loop();

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 1, &info);
    ASSERT_TRUE(message != NULL);
    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_msg_recv_nb(receiver->worker(), &recvbuf[0],
                                                recvbuf.size(), DATATYPE, message,
                                                recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
    request_release(my_recv_req);
}

UCS_TEST_F(test_ucp_tag, send_medium_msg_probe_truncated) {
    static const size_t size = 50000;
    ucp_tag_recv_info info;
    ucp_tag_message_h message;

    std::vector<char> sendbuf(size, 0);

    ucs::fill_random(sendbuf.begin(), sendbuf.end());

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    short_progress_loop();

    message = ucp_tag_probe_nb(receiver->worker(), 0x1337, 0xffff, 1, &info);
    ASSERT_TRUE(message != NULL);
    EXPECT_EQ(sendbuf.size(),      info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);

    request *my_recv_req;
    my_recv_req = (request*)ucp_tag_msg_recv_nb(receiver->worker(), NULL, 0,
                                                DATATYPE, message, recv_callback);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    wait(my_recv_req);

    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, my_recv_req->status);
    request_release(my_recv_req);
}

UCS_TEST_F(test_ucp_tag, cancel_exp) {
    uint64_t recv_data = 0;
    request *req;

    req = (request*)ucp_tag_recv_nb(receiver->worker(), &recv_data, sizeof(recv_data),
                                    DATATYPE, 1, 1, recv_callback);
    if (UCS_PTR_IS_ERR(req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(req));
    } else if (req == NULL) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
    }

    ucp_request_cancel(receiver->worker(), req);
    wait(req);

    EXPECT_EQ(UCS_ERR_CANCELED, req->status);
    EXPECT_EQ(0ul, recv_data);
    request_release(req);
}
