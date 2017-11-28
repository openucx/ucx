/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <vector>
#include <numeric>

#include "ucp_test.h"
#include <common/test_helpers.h>


class test_ucp_stream_base : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.field_mask  |= UCP_PARAM_FIELD_FEATURES;
        params.features     = UCP_FEATURE_STREAM;
        return params;
    }

    virtual void init() = 0;
    static void ucp_send_cb(void *request, ucs_status_t status) {}
    static void ucp_recv_cb(void *request, ucs_status_t status, size_t length) {}

    size_t wait_stream_recv(void *request);

protected:
    ucs_status_ptr_t stream_send_nb(const ucp::data_type_desc_t& dt_desc);
};

size_t test_ucp_stream_base::wait_stream_recv(void *request)
{
    ucs_status_t status;
    size_t       length;
    do {
        progress();
        status = ucp_stream_recv_request_test(request, &length);
    } while (status == UCS_INPROGRESS);
    ASSERT_UCS_OK(status);
    ucp_request_release(request);

    return length;
}

ucs_status_ptr_t
test_ucp_stream_base::stream_send_nb(const ucp::data_type_desc_t& dt_desc)
{
    return ucp_stream_send_nb(sender().ep(), dt_desc.buf(), dt_desc.count(),
                              dt_desc.dt(), ucp_send_cb, 0);
}

class test_ucp_stream : public test_ucp_stream_base
{
public:
    virtual void init() {
        ucp_test::init();

        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }
    }

protected:
    void do_send_recv_data_test(ucp_datatype_t datatype);
    void do_send_recv_test(ucp_datatype_t datatype);
    void do_send_exp_recv_test(ucp_datatype_t datatype);
};

void test_ucp_stream::do_send_recv_data_test(ucp_datatype_t datatype)
{
    std::vector<char> sbuf(size_t(16)*1024*1024, 's');
    size_t            ssize = 0; /* total send size */
    ucs_status_ptr_t  sstatus;
    std::vector<char> check_pattern;

    /* send all msg sizes*/
    for (size_t i = 3; i < sbuf.size(); i *= 2) {
        ucs::fill_random(sbuf.data(), i);
        check_pattern.insert(check_pattern.end(), sbuf.begin(), sbuf.begin() + i);
        sstatus = stream_send_nb(ucp::data_type_desc_t(datatype, sbuf.data(), i));
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
        ssize += i;
    }

    std::vector<char> rbuf(ssize, 'r');
    size_t            roffset = 0;
    ucs_status_ptr_t  rdata;
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
    EXPECT_EQ(check_pattern, rbuf);
}

void test_ucp_stream::do_send_recv_test(ucp_datatype_t datatype)
{
    std::vector<char> sbuf(size_t(16)*1024*1024, 's');
    size_t            ssize = 0; /* total send size */
    ucs_status_ptr_t  sstatus;
    std::vector<char> check_pattern;

    /* send all msg sizes*/
    for (size_t i = 3; i < sbuf.size(); i *= 2) {
        ucs::fill_random(sbuf.data(), i);
        check_pattern.insert(check_pattern.end(), sbuf.begin(), sbuf.begin() + i);
        sstatus = stream_send_nb(ucp::data_type_desc_t(datatype, sbuf.data(), i));
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
        ssize += i;
    }

    std::vector<char> rbuf(ssize, 'r');
    size_t            roffset = 0;
    do {
        ucp::data_type_desc_t dt_desc(datatype, &rbuf[roffset], ssize - roffset);

        size_t length;
        void   *rreq = ucp_stream_recv_nb(receiver().ep(), dt_desc.buf(),
                                          dt_desc.count(), dt_desc.dt(),
                                          ucp_recv_cb, &length, 0);
        if (UCS_PTR_IS_PTR(rreq)) {
            ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq));
            length = wait_stream_recv(rreq);
        }
        roffset += length;
    } while (roffset < ssize);

    EXPECT_EQ(roffset, ssize);
    EXPECT_EQ(check_pattern, rbuf);
}

void test_ucp_stream::do_send_exp_recv_test(ucp_datatype_t datatype)
{
    const size_t msg_size = size_t(4)*1024*1024;
    const size_t n_msgs   = 10;
    std::vector<std::vector<char> > rbufs(n_msgs,
                                          std::vector<char>(msg_size, 'r'));
    std::vector<ucp::data_type_desc_t> dt_rdescs(n_msgs);
    std::vector<void *> rreqs;

    /* post recvs */
    for (size_t i = 0; i < n_msgs; ++i) {
        ucp::data_type_desc_t &rdesc = dt_rdescs[i].make(datatype, &rbufs[i][0],
                                                         msg_size);
        size_t length;

        void *rreq = ucp_stream_recv_nb(receiver().ep(), rdesc.buf(),
                                        rdesc.count(), rdesc.dt(), ucp_recv_cb,
                                        &length, 0);
        EXPECT_TRUE(UCS_PTR_IS_PTR(rreq));
        rreqs.push_back(rreq);
    }

    std::vector<char>     sbuf(msg_size, 's');
    size_t                scount = 0; /* total send size */
    ucp::data_type_desc_t dt_desc(datatype, sbuf.data(), sbuf.size());

    /* send all msgs */
    for (size_t i = 0; i < n_msgs; ++i) {
        void *sreq = stream_send_nb(dt_desc);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sreq));
        wait(sreq);
        scount += sbuf.size();
    }

    size_t rcount = 0;
    for (size_t i = 0; i < rreqs.size(); ++i) {
        rcount += wait_stream_recv(rreqs[i]);
    }

    while (rcount < scount) {
        void             *buf   = &rbufs[0][0];
        size_t           count  = rbufs[0].size();
        size_t           length = std::numeric_limits<size_t>::max();
        ucs_status_ptr_t rreq;
        rreq = ucp_stream_recv_nb(receiver().ep(), buf, count, DATATYPE,
                                  ucp_recv_cb, &length, 0);
        if (UCS_PTR_IS_PTR(rreq)) {
            length = wait_stream_recv(rreq);
        }
        ASSERT_GT(length, 0);
        ASSERT_LE(length, msg_size);
        rcount += length;
    }
    EXPECT_EQ(scount, rcount);

    /* double check, no data should be here */
    while (progress());

    size_t s;
    void   *p;
    while ((p = ucp_stream_recv_data_nb(receiver().ep(), &s)) != NULL) {
        rcount += s;
        ucp_stream_data_release(receiver().ep(), p);
        progress();
    }
    EXPECT_EQ(scount, rcount);
}

UCS_TEST_P(test_ucp_stream, send_recv_data) {
    do_send_recv_data_test(DATATYPE);
}

UCS_TEST_P(test_ucp_stream, send_iov_recv_data) {
    do_send_recv_data_test(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream, send_recv) {
    do_send_recv_test(DATATYPE);
}

UCS_TEST_P(test_ucp_stream, send_recv_iov) {
    do_send_recv_test(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv) {
    do_send_exp_recv_test(DATATYPE);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_iov) {
    do_send_exp_recv_test(DATATYPE_IOV);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream)

class test_ucp_stream_many2one : public test_ucp_stream_base {
public:
    test_ucp_stream_many2one() : m_receiver_idx(3), m_nsenders(3) {
        m_recv_data.resize(m_nsenders);
    }

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

            ucp_ep_params_t recv_ep_param = get_ep_params();
            recv_ep_param.field_mask |= UCP_EP_PARAM_FIELD_USER_DATA;
            recv_ep_param.user_data   = (void *)uintptr_t(i);
            e(m_receiver_idx).connect(&e(i), recv_ep_param, i);
        }

        for (size_t i = 0; i < m_nsenders; ++i) {
            m_msgs.push_back(std::string("sender_") + ucs::to_string(i));
        }
    }

    static void ucp_send_cb(void *request, ucs_status_t status) {}
    static void ucp_recv_cb(void *request, ucs_status_t status, size_t length) {}

protected:
    static void erase_completed_reqs(std::vector<void *> &reqs);
    ucs_status_ptr_t stream_send_nb(size_t sender_idx, const void *buf,
                                    size_t count, ucp_datatype_t datatype);
    size_t send_all_nb(size_t n, std::vector<void *> &sreqs);
    size_t send_all(size_t n_iter);
    void check_no_data();
    void check_recv_data(size_t n_iter);

    std::vector<std::string>        m_msgs;
    std::vector<std::vector<char> > m_recv_data;
    const size_t                    m_receiver_idx;
    const size_t                    m_nsenders;
};

ucs_status_ptr_t
test_ucp_stream_many2one::stream_send_nb(size_t sender_idx, const void *buf,
                                         size_t count, ucp_datatype_t datatype)
{
    return ucp_stream_send_nb(m_entities.at(sender_idx).ep(),
                              buf, count, datatype, ucp_send_cb, 0);
}

size_t
test_ucp_stream_many2one::send_all_nb(size_t n_iter, std::vector<void *> &sreqs)
{
    size_t total = 0;
    /* Send many times in round robin */
    for (size_t i = 0; i < n_iter; ++i) {
        for (size_t sender_idx = 0; sender_idx < m_nsenders; ++sender_idx) {
            const void  *buf = m_msgs[sender_idx].c_str();
            size_t      len  = m_msgs[sender_idx].length();
            if (i == (n_iter - 1)) {
                ++len;
            }

            /* TODO: parameterize datatype */
            void *sreq = stream_send_nb(sender_idx, buf, len, DATATYPE);
            total += len;
            if (UCS_PTR_IS_PTR(sreq)) {
                sreqs.push_back(sreq);
            } else {
                EXPECT_FALSE(UCS_PTR_IS_ERR(sreq));
            }
        }
    }

    return total;
}

size_t
test_ucp_stream_many2one::send_all(size_t n_iter)
{
    std::vector<void *> sreqs;
    size_t              total;

    total = send_all_nb(n_iter, sreqs);
    while (!sreqs.empty()) {
        progress();
        erase_completed_reqs(sreqs);
    }

    return total;
}

void test_ucp_stream_many2one::check_no_data()
{
    const size_t         max_eps = 10;
    ucp_stream_poll_ep_t poll_eps[max_eps];
    ssize_t              count;

    ASSERT_EQ(m_receiver_idx, m_nsenders);
    for (size_t i = 0; i <= m_receiver_idx; ++i) {
        while(progress());
        count = ucp_stream_worker_poll(e(i).worker(), poll_eps, max_eps, 0);
        EXPECT_EQ(ssize_t(0), count);
    }
}

void test_ucp_stream_many2one::check_recv_data(size_t n_iter)
{
    for (size_t i = 0; i < m_nsenders; ++i) {
        const std::string test = std::string("sender_") + ucs::to_string(i);
        const std::string str(&m_recv_data[i].front());
        size_t            next  = 0;
        for (size_t j = 0; j < n_iter; ++j) {
            size_t match = str.find(test, next);
            EXPECT_NE(std::string::npos, match);
            if (match == std::string::npos) {
                break;
            }
            EXPECT_EQ(next, match);
            next += test.length();
        }
        EXPECT_EQ(next, str.length()); /* nothing more */
    }
}

void
test_ucp_stream_many2one::erase_completed_reqs(std::vector<void *> &reqs)
{
    std::vector<void *>::iterator i = reqs.begin();

    while (i != reqs.end()) {
        ucs_status_t status = ucp_request_check_status(*i);
        if (status != UCS_INPROGRESS) {
            EXPECT_EQ(UCS_OK, status);
            ucp_request_release(*i);
            i = reqs.erase(i);
        } else {
            ++i;
        }
    }
}

UCS_TEST_P(test_ucp_stream_many2one, drop_data) {
    send_all(10);

    ASSERT_EQ(m_receiver_idx, m_nsenders);
    for (size_t i = 0; i <= m_receiver_idx; ++i) {
        flush_worker(e(i));
    }

    /* Need to poll out all incoming data from transport layer */
    while (progress() > 0);
}

UCS_TEST_P(test_ucp_stream_many2one, worker_poll) {
    const size_t             niter = 2018;
    std::vector<void *>      sreqs;
    size_t                   total_len;

    total_len = send_all_nb(niter, sreqs);

    /* Recv and progress all data */
    do {
        ssize_t count;
        do {
            const size_t max_eps = 10;
            ucp_stream_poll_ep_t poll_eps[max_eps];
            progress();
            count = ucp_stream_worker_poll(e(m_receiver_idx).worker(),
                                           poll_eps, max_eps, 0);
            EXPECT_LE(0, count);

            for (ssize_t i = 0; i < count; ++i) {
                char   *rdata;
                size_t length;
                while ((rdata = (char *)ucp_stream_recv_data_nb(poll_eps[i].ep,
                                                                &length)) != NULL) {
                    ASSERT_FALSE(UCS_PTR_IS_ERR(rdata));
                    size_t senser_idx = uintptr_t(poll_eps[i].user_data);
                    std::vector<char> &dst = m_recv_data[senser_idx];
                    dst.insert(dst.end(), rdata, rdata + length);
                    total_len -= length;
                    ucp_stream_data_release(poll_eps[i].ep, rdata);
                }
            }
        } while (count > 0);

        erase_completed_reqs(sreqs);
    } while (!sreqs.empty() || (total_len != 0));

    check_no_data();
    check_recv_data(niter);
}

UCS_TEST_P(test_ucp_stream_many2one, recv_nb) {
    const size_t                            niter = 2018;
    std::vector<size_t>                     roffsets(m_nsenders, 0);
    std::vector<ucp::data_type_desc_t>      dt_rdescs(m_nsenders);
    std::vector<std::pair<size_t, void *> > rreqs;
    std::vector<void *>                     sreqs;
    size_t                                  total_sdata;

    ASSERT_FALSE(m_msgs.empty());

    /* Do preposts */
    for (size_t i = 0; i < m_nsenders; ++i) {
        m_recv_data[i].resize(m_msgs[i].length() * niter + 1);
        /* TODO: parameterize datatype */
        ucp::data_type_desc_t &rdesc = dt_rdescs[i].make(DATATYPE,
                                                         &m_recv_data[i][roffsets[i]],
                                                         m_recv_data[i].size());
        size_t length;
        void *rreq = ucp_stream_recv_nb(e(m_receiver_idx).ep(0, i),
                                        rdesc.buf(), rdesc.count(), rdesc.dt(),
                                        ucp_recv_cb, &length, 0);
        EXPECT_TRUE(UCS_PTR_IS_PTR(rreq));
        rreqs.push_back(std::make_pair(i, rreq));
    }

    total_sdata = send_all_nb(niter, sreqs);

    /* Recv and progress all the rest of data */
    do {
        ssize_t count;
        /* wait rreqs */
        for (size_t i = 0; i < rreqs.size(); ++i) {
            roffsets[rreqs[i].first] += wait_stream_recv(rreqs[i].second);
        }
        rreqs.clear();

        do {
            const size_t max_eps = 10;
            ucp_stream_poll_ep_t poll_eps[max_eps];
            progress();
            count = ucp_stream_worker_poll(e(m_receiver_idx).worker(),
                                           poll_eps, max_eps, 0);
            EXPECT_LE(0, count);

            for (ssize_t i = 0; i < count; ++i) {
                size_t sender_idx = uintptr_t(poll_eps[i].user_data);
                size_t &roffset   = roffsets[sender_idx];
                ucp::data_type_desc_t &dt_desc =
                    dt_rdescs[sender_idx].forward_to(roffset);
                EXPECT_TRUE(dt_desc.is_valid());
                size_t length;
                void *rreq = ucp_stream_recv_nb(poll_eps[i].ep, dt_desc.buf(),
                                                dt_desc.count(), dt_desc.dt(),
                                                ucp_recv_cb, &length, 0);
                if (UCS_PTR_STATUS(rreq) == UCS_OK) {
                    roffset += length;
                } else {
                    rreqs.push_back(std::make_pair(sender_idx, rreq));
                }
                EXPECT_FALSE(UCS_PTR_IS_ERR(rreq));
            }
        } while (count > 0);

        erase_completed_reqs(sreqs);
    } while (!rreqs.empty() || !sreqs.empty() ||
             (total_sdata > std::accumulate(roffsets.begin(),
                                            roffsets.end(), 0ul)));

    EXPECT_EQ(total_sdata, std::accumulate(roffsets.begin(),
                                           roffsets.end(), 0ul));
    check_no_data();
    check_recv_data(niter);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream_many2one)
