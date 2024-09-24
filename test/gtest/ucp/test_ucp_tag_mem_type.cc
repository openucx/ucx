/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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
        VARIANT_PROTO_V1    = UCS_BIT(2),
        VARIANT_MAX         = UCS_BIT(3)
    };

    void init()
    {
        int variant_flags = get_variant_value() / m_mem_type_pairs.size();

        if (variant_flags & VARIANT_GDR_OFF) {
            if (!has_any_transport(
                        {"dc_x", "ud_v", "ud_x", "rc_v", "rc_x", "ib"})) {
                UCS_TEST_SKIP_R("No GPU direct RDMA");
            }

            m_env.push_back(
                    new ucs::scoped_setenv("UCX_IB_GPU_DIRECT_RDMA", "n"));
        }

        if (variant_flags & VARIANT_TAG_OFFLOAD) {
            if (!has_any_transport({"rc_x", "dc_x", "ib"})) {
                UCS_TEST_SKIP_R("No tag offload");
            }

            enable_tag_mp_offload();

            if (RUNNING_ON_VALGRIND) {
                if (variant_flags & VARIANT_PROTO_V1) {
                    UCS_TEST_SKIP_R("Skip proto v1 with valgrind");
                }
                m_env.push_back(
                        new ucs::scoped_setenv("UCX_RC_TM_SEG_SIZE", "8k"));
                m_env.push_back(
                        new ucs::scoped_setenv("UCX_TCP_RX_SEG_SIZE", "8k"));
                m_env.push_back(
                        new ucs::scoped_setenv("UCX_RC_RX_QUEUE_LEN", "1024"));
            }
        }

        if (variant_flags & VARIANT_PROTO_V1) {
            modify_config("PROTO_ENABLE", "n");
        } else {
            modify_config("PROTO_REQUEST_RESET", "y");
        }

        int mem_type_pair_index = get_variant_value() % m_mem_type_pairs.size();
        m_send_mem_type         = m_mem_type_pairs[mem_type_pair_index][0];
        m_recv_mem_type         = m_mem_type_pairs[mem_type_pair_index][1];

        modify_config("MAX_EAGER_LANES", "2");
        modify_config("MAX_RNDV_LANES", "2");

        test_ucp_tag::init();
    }

    static void
    add_mem_type_test_variant(std::vector<ucp_test_variant> &variants,
                              int variant_value,
                              ucs_memory_type_t send_mem_type,
                              ucs_memory_type_t recv_mem_type)
    {
        std::string name = ucs_memory_type_names[send_mem_type] +
                           std::string(":") +
                           ucs_memory_type_names[recv_mem_type];

        int variant_flags = variant_value / m_mem_type_pairs.size();

        if (variant_flags & VARIANT_GDR_OFF) {
            if ((send_mem_type != UCS_MEMORY_TYPE_CUDA) &&
                (send_mem_type != UCS_MEMORY_TYPE_ROCM) &&
                (recv_mem_type != UCS_MEMORY_TYPE_CUDA) &&
                (recv_mem_type != UCS_MEMORY_TYPE_ROCM)) {
                /* No need to disable GPU-direct if the memory type does not
                   support it anyway */
                return;
            }
            name += ",nogdr";
        }

        if (variant_flags & VARIANT_TAG_OFFLOAD) {
            name += ",offload";
        }

        if (variant_flags & VARIANT_PROTO_V1) {
            name += ",proto_v1";
        }

        add_variant_with_value(variants, get_ctx_params(), variant_value, name);
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        int count = 0;
        for (int i = 0; i < VARIANT_MAX; i++) {
            for (const auto &mem_type_pair : m_mem_type_pairs) {
                add_mem_type_test_variant(variants, count, mem_type_pair[0],
                                          mem_type_pair[1]);
                ++count;
            }
        }
    }

    void do_basic_xfer(mem_buffer &send_buffer, mem_buffer &recv_buffer,
                       size_t length, ucs::detail::message_stream &ms)
    {
        const ucp_datatype_t type = ucp_dt_make_contig(1);

        ms << length << " " << std::flush;
        recv_buffer.pattern_fill(1, length);
        send_buffer.pattern_fill(2, length);
        size_t recvd = do_xfer(send_buffer.ptr(), recv_buffer.ptr(), length,
                               type, type, true, false, false);
        ASSERT_EQ(length, recvd);
        recv_buffer.pattern_check(2, length);
    }

    size_t max_test_length(unsigned exp) const
    {
        return static_cast<size_t>(pow(10.0, exp));
    }

    size_t test_length(unsigned exp) const
    {
        return (ucs::rand() % max_test_length(exp)) + 1;
    }

    static const
    std::vector<std::vector<ucs_memory_type_t> >& m_mem_type_pairs;

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

const std::vector<std::vector<ucs_memory_type_t> >&
test_ucp_tag_mem_type::m_mem_type_pairs = ucs::supported_mem_type_pairs();

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

UCS_TEST_P(test_ucp_tag_mem_type, realloc_buffers)
{
    std::vector<size_t> sizes =
            {0, 1, 16, 128, 1048512, 1011439, UCS_MBYTE + 4, 4194324};
    const size_t max_iter     = RUNNING_ON_VALGRIND ? 3 : 7;
    const size_t multiplier   = RUNNING_ON_VALGRIND ? 2 : 1;
    for (unsigned i = 0; i < max_iter; ++i) {
        sizes.push_back((i * multiplier));
    }

    ucs::detail::message_stream ms("INFO");
    for (auto length : sizes) {
        mem_buffer recv_mem_buf(length, m_recv_mem_type);
        mem_buffer send_mem_buf(length, m_send_mem_type);
        do_basic_xfer(send_mem_buf, recv_mem_buf, length, ms);
    }
}

// Set NUM_PATHS to 2 to allow multi-rail
UCS_TEST_P(test_ucp_tag_mem_type, reuse_buffers_mrail, "IB_NUM_PATHS?=2")
{
    const size_t max_length = max_test_length(7);
    mem_buffer recv_mem_buf(max_length, m_recv_mem_type);
    mem_buffer send_mem_buf(max_length, m_send_mem_type);

    // Test few specific sizes that expose corner cases, plush a few random ones
    std::vector<size_t> sizes = {0, 1, 16, 128, 1048512, UCS_MBYTE + 4, 4194324};
    const size_t max_iter     = RUNNING_ON_VALGRIND ? 1 : 4;
    for (unsigned i = 0; i < max_iter; ++i) {
        sizes.push_back(test_length(7));
    }

    ucs::detail::message_stream ms("INFO");
    for (auto length : sizes) {
        do_basic_xfer(send_mem_buf, recv_mem_buf, length, ms);
    }
}

UCS_TEST_P(test_ucp_tag_mem_type, rndv_4mb, "RNDV_THRESH=0")
{
    ucp_datatype_t type = ucp_dt_make_contig(1);
    const size_t length = 4 * UCS_MBYTE;

    mem_buffer recv_mem_buf(length, m_recv_mem_type, 1);
    mem_buffer send_mem_buf(length, m_send_mem_type, 2);

    size_t recvd = do_xfer(send_mem_buf.ptr(), recv_mem_buf.ptr(), length, type,
                           type, true, false, false);
    ASSERT_EQ(length, recvd);

    recv_mem_buf.pattern_check(2);
}

UCS_TEST_P(test_ucp_tag_mem_type, xfer_mismatch_length)
{
    ucp_datatype_t type = ucp_dt_make_contig(1);
    size_t length       = test_length(7);

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
