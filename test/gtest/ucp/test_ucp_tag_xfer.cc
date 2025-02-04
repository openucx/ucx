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
#include <ucp/core/ucp_resource.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/ptr_arith.h>
#include <uct/base/uct_iface.h>
}

#include <iostream>


class test_ucp_tag_xfer : public test_ucp_tag {
public:
    enum {
        VARIANT_DEFAULT,
        VARIANT_ERR_HANDLING,
        VARIANT_RNDV_PUT_ZCOPY,
        VARIANT_RNDV_GET_ZCOPY,
        VARIANT_RNDV_AM_BCOPY,
        VARIANT_RNDV_AM_ZCOPY,
        VARIANT_SEND_NBR,
        VARIANT_PROTO_V1
    };

    test_ucp_tag_xfer() {
        // TODO: test offload and offload MP as different variants
        enable_tag_mp_offload();

        if (RUNNING_ON_VALGRIND) {
            // Allow using TM MP offload for messages with a size of at least
            // 10000 bytes by setting HW TM segment size to 10 kB, since each
            // packet in TM MP offload is MTU-size buffer (i.e., in most cases
            // it is 4 kB segments)
            m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_SEG_SIZE", "10k"));
            m_env.push_back(new ucs::scoped_setenv("UCX_TCP_RX_SEG_SIZE", "8k"));
        }
    }

    virtual void init() {
        if (get_variant_value() == VARIANT_RNDV_PUT_ZCOPY) {
            modify_config("RNDV_SCHEME", "put_zcopy");
        } else if (get_variant_value() == VARIANT_RNDV_GET_ZCOPY) {
            modify_config("RNDV_SCHEME", "get_zcopy");
        } else if (get_variant_value() == VARIANT_RNDV_AM_BCOPY) {
            modify_config("RNDV_SCHEME", "am");
            modify_config("ZCOPY_THRESH", "inf");
        } else if (get_variant_value() == VARIANT_RNDV_AM_ZCOPY) {
            modify_config("RNDV_SCHEME", "am");
            modify_config("ZCOPY_THRESH", "0");
        } else if (get_variant_value() == VARIANT_PROTO_V1) {
            modify_config("PROTO_ENABLE", "n");
        }

        /* Init number of lanes according to test requirement
         * (default is 2, for max_lanes test we use max_lanes) */
        std::string num_lanes_str(std::to_string(num_lanes()));

        modify_config("MAX_EAGER_LANES", num_lanes_str);
        modify_config("MAX_RNDV_LANES", num_lanes_str);

        test_ucp_tag::init();
    }

    virtual void cleanup()
    {
        EXPECT_EQ(ucp::dt_gen_start_count, ucp::dt_gen_finish_count);
        test_ucp_tag::cleanup();
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant_with_value(variants, get_ctx_params(), VARIANT_DEFAULT, "");
        add_variant_with_value(variants, get_ctx_params(),
                               VARIANT_ERR_HANDLING, "err_handling");
        add_variant_with_value(variants, get_ctx_params(),
                               VARIANT_RNDV_PUT_ZCOPY, "rndv_put_zcopy");
        add_variant_with_value(variants, get_ctx_params(),
                               VARIANT_RNDV_GET_ZCOPY, "rndv_get_zcopy");
        add_variant_with_value(variants, get_ctx_params(),
                               VARIANT_RNDV_AM_BCOPY, "rndv_am_bcopy");
        add_variant_with_value(variants, get_ctx_params(),
                               VARIANT_RNDV_AM_ZCOPY, "rndv_am_zcopy");
        add_variant_with_value(variants, get_ctx_params(),
                               VARIANT_SEND_NBR, "send_nbr");
        if (!RUNNING_ON_VALGRIND) {
            add_variant_with_value(variants, get_ctx_params(), VARIANT_PROTO_V1,
                                   "proto_v1");
        }
    }

    virtual ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t ep_params = test_ucp_tag::get_ep_params();
        if (get_variant_value() == VARIANT_ERR_HANDLING) {
            ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
            ep_params.err_mode    = UCP_ERR_HANDLING_MODE_PEER;
        }
        return ep_params;
    }

    bool is_err_handling() const {
        return get_variant_value() == VARIANT_ERR_HANDLING;
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

    size_t do_xfer(const void *sendbuf, void *recvbuf, size_t count,
                   ucp_datatype_t send_dt, ucp_datatype_t recv_dt,
                   bool expected, bool sync, bool truncated);

    void test_xfer(xfer_func_t func, bool expected, bool sync, bool truncated);
    void test_run_xfer(bool send_contig, bool recv_contig,
                       bool expected, bool sync, bool truncated);
    void test_xfer_prepare_bufs(uint8_t *sendbuf, uint8_t *recvbuf, size_t count,
                                bool send_contig, bool recv_contig,
                                ucp_datatype_t *send_dt,
                                ucp_datatype_t *recv_dt);
    void test_xfer_probe(bool send_contig, bool recv_contig,
                         bool expected, bool sync);

    void test_xfer_len_offset();
    virtual size_t get_msg_size();

    /* Init number of lanes which will be used */
    virtual unsigned num_lanes()
    {
        return 2;
    }

private:
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
        status = ucp_dt_create_generic(&ucp::test_dt_uint8_ops, NULL, send_dt);
        ASSERT_UCS_OK(status);
    }

    if (recv_contig) {
        /* the recv has a contig datatype for the data buffer */
        *recv_dt = DATATYPE;
    } else {
        /* the receiver has a generic datatype */
        status = ucp_dt_create_generic(&ucp::test_dt_uint8_ops, NULL, recv_dt);
        /* the recvbuf can be NULL because we only validate the received data in the
        * unpack function - we don't copy it to the recvbuf */
        ASSERT_UCS_OK(status);
    }
}

size_t test_ucp_tag_xfer::get_msg_size()
{
    size_t raw_size = 1148544 / ucs::test_time_multiplier();
    return ucs_align_up(raw_size, ucs_get_page_size() * num_lanes());
}

void test_ucp_tag_xfer::test_run_xfer(bool send_contig, bool recv_contig,
                                      bool expected, bool sync, bool truncated)
{
    const size_t msg_size = get_msg_size();
    uint8_t *sendbuf = NULL, *recvbuf = NULL;
    ucp_datatype_t send_dt, recv_dt;
    size_t recvd;

    if (sync) {
        skip_err_handling();
    }

    ucp::dt_gen_start_count  = 0;
    ucp::dt_gen_finish_count = 0;

    if (send_contig) {
        /* the sender has a contig datatype for the data buffer */
        sendbuf = (uint8_t*)aligned_alloc(ucs_get_page_size(),
                                          msg_size * sizeof(*sendbuf));
    }
    if (recv_contig) {
        /* the recv has a contig datatype for the data buffer */
        recvbuf = (uint8_t*)aligned_alloc(ucs_get_page_size(),
                                          msg_size * sizeof(*recvbuf));
    }

    test_xfer_prepare_bufs(sendbuf, recvbuf, msg_size, send_contig, recv_contig,
                           &send_dt, &recv_dt);

    /* coverity[var_deref_model] */
    /* coverity[var_deref_op] */
    recvd = do_xfer(&sendbuf[0], &recvbuf[0], msg_size, send_dt, recv_dt,
                    expected, sync, truncated);
    if (!truncated) {
        EXPECT_EQ(msg_size * sizeof(uint8_t), recvd);
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

    /* the self transport doesn't do rndv and completes the send immediately */
    skip_loopback();

    ucp::dt_gen_start_count  = 0;
    ucp::dt_gen_finish_count = 0;

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
        request_free(sreq);
    }
    request_free(rreq);

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

    ucp::dt_gen_start_count  = 0;
    ucp::dt_gen_finish_count = 0;

    /* if count is zero, truncation has no effect */
    if ((truncated) && (!count)) {
        truncated = false;
    }

    status = ucp_dt_create_generic(&ucp::test_dt_uint32_ops, NULL, &dt);
    ASSERT_UCS_OK(status);

    recvd = do_xfer(NULL, NULL, count, dt, dt, expected, sync, truncated);
    if (!truncated) {
        EXPECT_EQ(count * sizeof(uint32_t), recvd);
    }

    /* restart send request might call extra start/finish */
    EXPECT_GE(ucp::dt_gen_start_count, 2);
    EXPECT_GE(ucp::dt_gen_finish_count, 2);
    EXPECT_EQ(ucp::dt_gen_start_count, ucp::dt_gen_finish_count);

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

    ucp::dt_gen_start_count  = 0;
    ucp::dt_gen_finish_count = 0;

    status = ucp_dt_create_generic(&ucp::test_dt_uint32_err_ops, this, &dt);
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
        request_free(sreq);
    }

    if (count > 0) {
        /* the generic unpack function is expected to fail */
        EXPECT_EQ(UCS_ERR_NO_MEMORY, rreq->status);
    }
    request_free(rreq);

    /* restart send request might call extra start/finish */
    EXPECT_GE(ucp::dt_gen_start_count, 2);
    EXPECT_GE(ucp::dt_gen_finish_count, 2);
    EXPECT_EQ(ucp::dt_gen_start_count, ucp::dt_gen_finish_count);
    ucp_dt_destroy(dt);
}

test_ucp_tag_xfer::request*
test_ucp_tag_xfer::do_send(const void *sendbuf, size_t count, ucp_datatype_t dt,
                           bool sync)
{
    if (sync) {
        return send_sync_nb(sendbuf, count, dt, SENDER_TAG);
    } else {
        if (get_variant_value() == VARIANT_SEND_NBR) {
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
        request_free(sreq);
    }

    recvd = rreq->info.length;
    if (!truncated) {
        EXPECT_UCS_OK(rreq->status);
        EXPECT_EQ((ucp_tag_t)SENDER_TAG, rreq->info.sender_tag);
    } else {
        EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, rreq->status);
    }

    request_free(rreq);
    return recvd;
}

void test_ucp_tag_xfer::test_xfer_len_offset()
{
    const size_t max_offset  = 128;
    const size_t max_length  = 64 * UCS_KBYTE;
    const size_t min_length  = UCS_KBYTE;
    const size_t offset_step = 16;
    const size_t length_step = 16;
    const size_t buf_size    = max_length + max_offset + 2;
    ucp_datatype_t type      = ucp_dt_make_contig(1);
    void *send_buf           = 0;
    void *recv_buf           = 0;
    size_t offset;
    size_t length;
    ucs::detail::message_stream *ms;

    skip_err_handling();

    EXPECT_EQ(posix_memalign(&send_buf, 8192, buf_size), 0);
    EXPECT_EQ(posix_memalign(&recv_buf, 8192, buf_size), 0);

    memset(send_buf, 0, buf_size);
    memset(recv_buf, 0, buf_size);

    for (offset = 0; offset <= max_offset; offset += offset_step) {
        if (!offset || ucs_is_pow2(offset)) {
            ms = new ucs::detail::message_stream("INFO");
            *ms << "offset: " << offset << ": ";
        } else {
            ms = NULL;
        }
        for (length = min_length; length <= max_length; length += length_step) {
            if (ms && ucs_is_pow2(length)) {
                *ms << length << " ";
                fflush(stdout);
            }

            do_xfer((char*)send_buf + offset, (char*)recv_buf + offset,
                    length, type, type, true, true, false);
            do_xfer((char*)send_buf + max_offset - offset,
                    (char*)recv_buf + max_offset - offset,
                    length, type, type, true, true, false);
        }
        if (ms) {
            delete(ms);
        }
    }

    free(recv_buf);
    free(send_buf);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_truncated) {
    check_offload_support(false);
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

UCS_TEST_P(test_ucp_tag_xfer, generic_unexp_truncated) {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic, false, false, true);
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

UCS_TEST_P(test_ucp_tag_xfer, generic_err_exp, "PROTO_INDIRECT_ID=y") {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, true, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_unexp, "PROTO_INDIRECT_ID=y") {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, false, false, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_exp_sync, "PROTO_INDIRECT_ID=y") {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, generic_err_unexp_sync, "PROTO_INDIRECT_ID=y") {
    test_xfer(&test_ucp_tag_xfer::test_xfer_generic_err, false, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_sync) {
    /* because ucp_tag_send_req return status (instead request) if send operation
     * completed immediately */
    skip_loopback();
    test_xfer(&test_ucp_tag_xfer::test_xfer_contig, true, true, false);
}

UCS_TEST_P(test_ucp_tag_xfer, contig_exp_sync_zcopy, "ZCOPY_THRESH=1000") {
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
                                                                          "ZCOPY_THRESH=1248576") {
    check_offload_support(false);
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
    test_run_xfer(true, true, true, true, true);
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
    test_run_xfer(true, true, false, true, true);
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
    test_run_xfer(false, false, true, true, true);
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
    test_run_xfer(false, false, false, true, true);
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
    test_run_xfer(false, true, true, true, true);
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
    test_run_xfer(false, true, false, true, true);
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

UCS_TEST_SKIP_COND_P(test_ucp_tag_xfer, test_xfer_len_offset,
                     RUNNING_ON_VALGRIND, "RNDV_THRESH=1000") {
    test_xfer_len_offset();
}

UCS_TEST_P(test_ucp_tag_xfer, iov_with_empty_buffers, "ZCOPY_THRESH=512") {
    const size_t iovcnt    = ucp::data_type_desc_t::MAX_IOV;
    const size_t size      = UCS_KBYTE;
    const int    expected  = 1;
    const int    sync      = 0;
    const int    truncated = 0;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);
    ucp_dt_iov_t iovec[iovcnt];

    ucs::fill_random(sendbuf);

    /* initialize iovec with MAX_IOV-1 empty buffers and one non-empty */
    for (size_t i = 0; i < iovcnt - 1; ++i) {
        iovec[i].buffer = NULL;
        iovec[i].length = 0;
    }

    /* coverity[escape] */
    iovec[iovcnt - 1].buffer = &sendbuf[0];
    iovec[iovcnt - 1].length = size;

    ucp::data_type_desc_t recv_dt_desc(DATATYPE_IOV, recvbuf.data(),
                                       recvbuf.size(), iovcnt);

    size_t recvd = do_xfer(iovec, recv_dt_desc.buf(), iovcnt,
                           DATATYPE_IOV, DATATYPE_IOV, expected, 0,
                           truncated);

    ASSERT_EQ(sendbuf.size(), recvd);
    EXPECT_TRUE(!check_buffers(sendbuf, recvbuf, recvd, iovcnt,
                               recv_dt_desc.count(), size, expected, sync,
                               "IOV"));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_xfer)


#ifdef ENABLE_STATS

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

    using test_ucp_tag::get_test_variants;

    ucs_stats_node_t* ep_stats(entity &e) {
        return e.ep()->stats;
    }

    ucs_stats_node_t* worker_stats(entity &e) {
        return e.worker()->stats;
    }

    unsigned get_tx_stat(unsigned counter) {
        return UCS_STATS_GET_COUNTER(worker_stats(sender()), counter);
    }

    unsigned get_rx_stat(unsigned counter) {
        return UCS_STATS_GET_COUNTER(worker_stats(receiver()), counter);
    }

    void validate_counters(unsigned tx_counter, unsigned rx_counter) {
        uint64_t cnt;
        cnt = UCS_STATS_GET_COUNTER(ep_stats(sender()), tx_counter);
        EXPECT_EQ(1ul, cnt) << "TX counter";
        cnt = get_rx_stat(rx_counter);
        EXPECT_EQ(1ul, cnt) << "RX counter";
    }

    bool has_xpmem() {
        return ucp_context_find_tl_md(receiver().ucph(), "xpmem") != NULL;
    }

    bool has_rma_zcopy() {
        return has_transport("rc_v") || has_transport("rc_x") ||
               has_transport("dc_x") ||
               (ucp_context_find_tl_md(receiver().ucph(), "cma")  != NULL) ||
               (ucp_context_find_tl_md(receiver().ucph(), "knem") != NULL);
    }

    void validate_rndv_counters() {
        unsigned get_zcopy = get_rx_stat(UCP_WORKER_STAT_RNDV_GET_ZCOPY);
        unsigned put_zcopy = get_tx_stat(UCP_WORKER_STAT_RNDV_PUT_ZCOPY);
        unsigned rtr       = get_rx_stat(UCP_WORKER_STAT_RNDV_RTR);
        unsigned rkey_ptr  = get_rx_stat(UCP_WORKER_STAT_RNDV_RKEY_PTR);

        UCS_TEST_MESSAGE << "get_zcopy: " << get_zcopy << " rtr: " << rtr
                         << " rkey_ptr: " << rkey_ptr;
        EXPECT_EQ(1, get_zcopy + rtr + rkey_ptr);
        EXPECT_LE(put_zcopy, rtr);

        if (is_proto_enabled()) {
            if (has_xpmem() || is_self() || has_rma_zcopy()) {
                /* Expect one of the bulk transfer protocols */
                EXPECT_EQ(1u, rkey_ptr + get_zcopy + put_zcopy);
                return;
            }
        } else {
            if (has_xpmem() || is_self()) {
                /* rkey_ptr expected to be selected if xpmem is available or
                   self is being used */
                EXPECT_EQ(1u, rkey_ptr);
                return;
            }

            if (has_rma_zcopy()) {
                /* If any transports supports get_zcopy, expect it to be used */
                EXPECT_EQ(1u, get_zcopy);
                return;
            }
        }

        /* Could be a transport which supports get_zcopy that wasn't accounted
         * for, or fallback to RTR. Anyway, rkey_ptr is not expected to be used.
         */
        EXPECT_EQ(1u, rtr + get_zcopy);
    }
};


UCS_TEST_P(test_ucp_tag_stats, eager_expected, "RNDV_THRESH=1248576") {
    check_offload_support(false);
    test_run_xfer(true, true, true, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER,
                      UCP_WORKER_STAT_TAG_RX_EAGER_MSG);

    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_EQ(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, eager_unexpected, "RNDV_THRESH=1248576") {
    check_offload_support(false);
    test_run_xfer(true, true, false, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER,
                      UCP_WORKER_STAT_TAG_RX_EAGER_MSG);
    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_GT(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, sync_expected, "RNDV_THRESH=1248576") {
    check_offload_support(false);
    skip_loopback();
    test_run_xfer(true, true, true, true, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER_SYNC,
                      UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG);

    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                 UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_EQ(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, sync_unexpected, "RNDV_THRESH=1248576") {
    check_offload_support(false);
    skip_loopback();
    test_run_xfer(true, true, false, true, false);
    validate_counters(UCP_EP_STAT_TAG_TX_EAGER_SYNC,
                      UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG);
    uint64_t cnt;
    cnt = UCS_STATS_GET_COUNTER(worker_stats(receiver()),
                                UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP);
    EXPECT_GT(cnt, 0ul);
}

UCS_TEST_P(test_ucp_tag_stats, rndv_expected, "RNDV_THRESH=1000") {
    check_offload_support(false);
    test_run_xfer(true, true, true, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_RNDV, UCP_WORKER_STAT_RNDV_RX_EXP);
    validate_rndv_counters();
}

UCS_TEST_P(test_ucp_tag_stats, rndv_unexpected, "RNDV_THRESH=1000") {
    check_offload_support(false);
    test_run_xfer(true, true, false, false, false);
    validate_counters(UCP_EP_STAT_TAG_TX_RNDV, UCP_WORKER_STAT_RNDV_RX_UNEXP);
    validate_rndv_counters();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_stats)

class multi_rail_max : public test_ucp_tag_xfer {
public:
    void init() override
    {
        stats_activate();
        test_ucp_tag_xfer::init();
    }

    size_t get_msg_size() override
    {
        return ucs_align_up(10 * UCS_MBYTE, ucs_get_page_size() * num_lanes());
    }

    void cleanup() override
    {
        test_ucp_tag_xfer::cleanup();
        stats_restore();
    }

    uint64_t get_bytes_sent(ucp_ep_h ep, int lane_index)
    {
        uct_base_ep_t *uct_ep = ucs_derived_of(ucp_ep_get_lane(ep, lane_index),
                                               uct_base_ep_t);

        return UCS_STATS_GET_COUNTER(uct_ep->stats, UCT_EP_STAT_BYTES_SHORT) +
               UCS_STATS_GET_COUNTER(uct_ep->stats, UCT_EP_STAT_BYTES_BCOPY) +
               UCS_STATS_GET_COUNTER(uct_ep->stats, UCT_EP_STAT_BYTES_ZCOPY);
    }

    unsigned num_lanes() override
    {
        return UCP_MAX_LANES;
    }

};

UCS_TEST_P(multi_rail_max, max_lanes, "IB_NUM_PATHS?=64", "TM_SW_RNDV=y",
           "RNDV_THRESH=1", "MIN_RNDV_CHUNK_SIZE=1", "MULTI_PATH_RATIO=0.0001")
{
    receiver().connect(&sender(), get_ep_params());
    test_run_xfer(true, true, true, true, false);

    EXPECT_EQ(num_lanes(), ucp_ep_num_lanes(sender().ep()));
    EXPECT_EQ(num_lanes(), ucp_ep_num_lanes(receiver().ep()));

    size_t bytes_sent = 0;
    for (ucp_lane_index_t lane = 0; lane < num_lanes(); ++lane) {
        size_t sender_tx   = get_bytes_sent(sender().ep(), lane);
        size_t receiver_tx = get_bytes_sent(receiver().ep(), lane);
        UCS_TEST_MESSAGE << "lane[" << static_cast<int>(lane) << "] : "
                         << "sender " << sender_tx << " receiver " << receiver_tx;

        EXPECT_GT(sender_tx + receiver_tx, 0);
        bytes_sent += sender_tx + receiver_tx;
    }

    EXPECT_GE(bytes_sent, get_msg_size());
}

UCP_INSTANTIATE_TEST_CASE_TLS(multi_rail_max, rc, "rc")

#endif
