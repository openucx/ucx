/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
#include <common/mem_buffer.h>

#include "test_ucp_tag.h"
#include "ucp_datatype.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <ucs/datastruct/queue.h>
}

#include <iostream>


class test_ucp_tag_mem_type: public test_ucp_tag {
public:
    enum {
        VARIANT_GDR_OFF     = UCS_BIT(0),
        VARIANT_TAG_OFFLOAD = UCS_BIT(1),
        VARIANT_MAX         = UCS_BIT(2)
    };

    void init() {
        int mem_type_pair_index = get_variant_value() % mem_type_pairs.size();
        int varient_index       = get_variant_value() / mem_type_pairs.size();

        if (varient_index & VARIANT_GDR_OFF) {
            m_env.push_back(new ucs::scoped_setenv("UCX_IB_GPU_DIRECT_RDMA", "n"));
        }

        if (varient_index & VARIANT_TAG_OFFLOAD) {
            enable_tag_mp_offload();

            if (RUNNING_ON_VALGRIND) {
                m_env.push_back(new ucs::scoped_setenv("UCX_RC_TM_SEG_SIZE",  "8k"));
                m_env.push_back(new ucs::scoped_setenv("UCX_TCP_RX_SEG_SIZE", "8k"));
            }
        }

        m_send_mem_type  = mem_type_pairs[mem_type_pair_index][0];
        m_recv_mem_type  = mem_type_pairs[mem_type_pair_index][1];

        modify_config("MAX_EAGER_LANES", "2");
        modify_config("MAX_RNDV_LANES",  "2");

        test_ucp_tag::init();
    }

    void cleanup() {
        test_ucp_tag::cleanup();
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        int count = 0;
        for (int i = 0; i < VARIANT_MAX; i++) {
            for (std::vector<std::vector<ucs_memory_type_t> >::const_iterator iter =
                 mem_type_pairs.begin(); iter != mem_type_pairs.end(); ++iter) {
                std::string name =
                        std::string(ucs_memory_type_names[(*iter)[0]]) + ":" +
                        std::string(ucs_memory_type_names[(*iter)[1]]);
                if (i & VARIANT_GDR_OFF) {
                    name += ",nogdr";
                }
                if (i & VARIANT_TAG_OFFLOAD) {
                    name += ",offload";
                }
                add_variant_with_value(variants, get_ctx_params(), count, name);
                ++count;
            }
        }
    }

    void do_basic_send(void *sbuf, void *rbuf, size_t length, ucp_datatype_t type,
                       ucs_memory_type_t s_mem_type, ucs_memory_type_t r_mem_type)
    {
        mem_buffer::pattern_fill(rbuf, length, 1, r_mem_type);
        mem_buffer::pattern_fill(sbuf, length, 2, s_mem_type);
        size_t recvd = do_xfer(sbuf, rbuf, length, type, type, true, false, false);
        ASSERT_EQ(length, recvd);
        mem_buffer::pattern_check(rbuf, length, 2, r_mem_type);
    }

    static std::vector<std::vector<ucs_memory_type_t> > mem_type_pairs;

protected:

    size_t do_xfer(const void *sendbuf, void *recvbuf, size_t count,
                   ucp_datatype_t send_dt, ucp_datatype_t recv_dt,
                   bool expected, bool truncated, bool extended);

    ucs_memory_type_t m_send_mem_type;
    ucs_memory_type_t m_recv_mem_type;

private:

    static const uint64_t SENDER_TAG = 0x111337;
    static const uint64_t RECV_MASK  = 0xffff;
    static const uint64_t RECV_TAG   = 0x1337;
};

std::vector<std::vector<ucs_memory_type_t> >
test_ucp_tag_mem_type::mem_type_pairs = ucs::supported_mem_type_pairs();

size_t test_ucp_tag_mem_type::do_xfer(const void *sendbuf, void *recvbuf,
                                  size_t count, ucp_datatype_t send_dt,
                                  ucp_datatype_t recv_dt, bool expected,
                                  bool truncated, bool extended)
{
    size_t recv_count = count;
    size_t send_count = count;
    size_t recvd      = 0;
    request *rreq, *sreq;

    if (truncated) {
        recv_count /= 2;
    }

    if (extended) {
        send_count /= 2;
    }

    if (expected) {
        rreq = recv_nb(recvbuf, recv_count, recv_dt, RECV_TAG, RECV_MASK);
        sreq = send_nb(sendbuf, send_count, send_dt, SENDER_TAG);
    } else {
        sreq = send_nb(sendbuf, send_count, send_dt, SENDER_TAG);

        wait_for_unexpected_msg(receiver().worker(), 10.0);

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
};

UCS_TEST_P(test_ucp_tag_mem_type, basic)
{
    ucp_datatype_t type = ucp_dt_make_contig(1);
    size_t max_length;

    UCS_TEST_MESSAGE << "TEST: "
                     << ucs_memory_type_names[m_send_mem_type] << " <-> "
                     << ucs_memory_type_names[m_recv_mem_type];

    for (unsigned i = 1; i <= 7; ++i) {
        max_length = (size_t)pow(10.0, i);
        size_t length = ucs::rand() % max_length + 1;

        mem_buffer m_recv_mem_buf(length, m_recv_mem_type);
        mem_buffer m_send_mem_buf(length, m_send_mem_type);

        do_basic_send(m_send_mem_buf.ptr(),m_recv_mem_buf.ptr(), length, type,
                      m_send_mem_buf.mem_type(), m_recv_mem_buf.mem_type());
    }

    /*  test with re-using the buffers */
    max_length = (size_t)pow(10.0, 7);
    mem_buffer m_recv_mem_buf(max_length, m_recv_mem_type);
    mem_buffer m_send_mem_buf(max_length, m_send_mem_type);

    for (unsigned i = 0; i < 2; ++i) {
        size_t length = ucs::rand() % max_length + 1;

        do_basic_send(m_send_mem_buf.ptr(),m_recv_mem_buf.ptr(), length, type,
                      m_send_mem_buf.mem_type(), m_recv_mem_buf.mem_type());
    }

}

UCS_TEST_P(test_ucp_tag_mem_type, xfer_mismatch_length)
{
    ucp_datatype_t type = ucp_dt_make_contig(1);
    size_t length = ucs::rand() % ((ssize_t)pow(10.0, 7));

    UCS_TEST_MESSAGE << "TEST: "
                     << ucs_memory_type_names[m_send_mem_type] << " <-> "
                     << ucs_memory_type_names[m_recv_mem_type] << " length: "
                     << length;

    mem_buffer m_recv_mem_buf(length, m_recv_mem_type);
    mem_buffer m_send_mem_buf(length, m_send_mem_type);

    mem_buffer::pattern_fill(m_recv_mem_buf.ptr(), m_recv_mem_buf.size(),
                             1, m_recv_mem_buf.mem_type());

    mem_buffer::pattern_fill(m_send_mem_buf.ptr(), m_send_mem_buf.size(),
                             2, m_send_mem_buf.mem_type());

    /* truncated */
    do_xfer(m_send_mem_buf.ptr(), m_recv_mem_buf.ptr(),
            length, type, type, true, true, false);

    /* extended recv buffer */
    size_t recvd = do_xfer(m_send_mem_buf.ptr(), m_recv_mem_buf.ptr(),
                           length, type, type, true, false, true);
    ASSERT_EQ(length / 2,  recvd);

}


UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_tag_mem_type);
