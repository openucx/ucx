/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"


class test_ucp_stream : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_STREAM;
        return params;
    }

    virtual void init() {
        ucp_test::init();

        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }
    }

    static void ucp_send_cb(void *request, ucs_status_t status) {}
};

UCS_TEST_P(test_ucp_stream, send_recv_data) {
    std::vector<char> sbuf(size_t(16)*1024*1024, 's');
    size_t            ssize = 0; /* total send size */

    /* send all msg sizes*/
    for (size_t i = 3; i < sbuf.size(); i *= 2) {
        ucs_status_ptr_t sstatus = ucp_stream_send_nb(sender().ep(), sbuf.data(),
                                                      i, ucp_dt_make_contig(1),
                                                      ucp_send_cb, 0);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
        ssize += i;
    }

    std::vector<char> rbuf(ssize, 'r');
    size_t            roffset = 0;
    ucs_status_ptr_t rdata;
    size_t length;
    do {
        progress();
        rdata = ucp_stream_recv_data_nb(receiver().ep(), &length);
        if (UCS_PTR_STATUS(rdata) == UCS_OK) {
            continue;
        }

        memcpy(&rbuf[roffset], rdata, length);
        roffset += length;
        ucp_stream_data_release(receiver().ep(), rdata);
    } while (roffset < ssize);

    EXPECT_EQ(roffset, ssize);
    sbuf.resize(ssize, 's');
    EXPECT_EQ(sbuf, rbuf);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream)

class test_ucp_stream_many2one : public ucp_test {
public:
    test_ucp_stream_many2one() : m_receiver_idx(3), m_nsenders(3) {}

    static ucp_params_t get_ctx_params() {
        return test_ucp_stream::get_ctx_params();
    }

    virtual void init() {
        /* Skip entities creation */
        test_base::init();

        if (is_self()) {
            UCS_TEST_SKIP_R("self");
        }

        for (size_t i = 0; i < m_nsenders + 1; ++i) {
            create_entity();
        }

        for (size_t i = 0; i < m_nsenders; ++i) {
            e(i).connect(&e(m_receiver_idx), get_ep_params(), i);
            e(m_receiver_idx).connect(&e(i), get_ep_params(), i);
        }

        for (size_t i = 0; i < m_nsenders; ++i) {
            msgs.push_back(std::string("sender_") + ucs::to_string(i));
        }
    }

    static void ucp_send_cb(void *request, ucs_status_t status) {}

protected:
    ucs_status_ptr_t stream_send_nb(size_t sender_idx, const void *buf,
                                    size_t count, ucp_datatype_t datatype);

    std::vector<std::string>    msgs;
    const size_t                m_receiver_idx;
    const size_t                m_nsenders;
};

ucs_status_ptr_t
test_ucp_stream_many2one::stream_send_nb(size_t sender_idx, const void *buf,
                                         size_t count, ucp_datatype_t datatype)
{
    return ucp_stream_send_nb(m_entities.at(sender_idx).ep(),
                              buf, count, datatype, ucp_send_cb, 0);
}

UCS_TEST_P(test_ucp_stream_many2one, drop_data) {
    for (size_t sender_idx = 0; sender_idx < m_nsenders; ++sender_idx) {
        const void  *buf = reinterpret_cast<const void *>(msgs[sender_idx].c_str());
        size_t      len  = msgs[sender_idx].length() + 1;

        ucs_status_ptr_t sstatus = stream_send_nb(sender_idx, buf, len,
                                                  ucp_dt_make_contig(1));
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
    }

    for (size_t i = 0; i < m_nsenders + 1; ++i) {
        e(i).flush_worker();
    }

    /* Need to poll out all incoming data from transport layer */
    while (progress() > 0);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream_many2one)
