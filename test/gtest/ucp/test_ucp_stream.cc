/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <list>
#include <numeric>
#include <set>
#include <vector>

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
    template <typename T>
    void do_send_recv_test(ucp_datatype_t datatype);
    template <typename T>
    void do_send_exp_recv_test(ucp_datatype_t datatype);
    void do_send_recv_data_recv_test(ucp_datatype_t datatype);
};

void test_ucp_stream::do_send_recv_data_test(ucp_datatype_t datatype)
{
    std::vector<char> sbuf(16 * 1024 * 1024, 's');
    size_t            ssize = 0; /* total send size in bytes */
    std::vector<char> check_pattern;
    ucs_status_ptr_t  sstatus;

    /* send all msg sizes*/
    for (size_t i = 3; i < sbuf.size(); i *= 2) {
        ucs::fill_random(sbuf, i);
        check_pattern.insert(check_pattern.end(), sbuf.begin(),
                             sbuf.begin() + i);
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

template <typename T>
void test_ucp_stream::do_send_recv_test(ucp_datatype_t datatype)
{
    const size_t      dt_elem_size = UCP_DT_IS_CONTIG(datatype) ?
                                     ucp_contig_dt_elem_size(datatype) : 1;
    std::vector<char> sbuf(16 * 1024 * 1024, 's');
    size_t            ssize = 0; /* total send size */
    ucs_status_ptr_t  sstatus;
    std::vector<char> check_pattern;

    /* send all msg sizes in bytes*/
    for (size_t i = 3; i < sbuf.size(); i *= 2) {
        ucs::fill_random(sbuf, i);
        check_pattern.insert(check_pattern.end(), sbuf.begin(), sbuf.begin() + i);
        sstatus = stream_send_nb(ucp::data_type_desc_t(DATATYPE, sbuf.data(), i));
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
        ssize += i;
    }

    size_t align_tail = dt_elem_size - ssize % dt_elem_size;
    if (align_tail != 0) {
        ucs::fill_random(sbuf, align_tail);
        check_pattern.insert(check_pattern.end(), sbuf.begin(), sbuf.begin() + align_tail);
        sstatus = stream_send_nb(ucp::data_type_desc_t(ucp_dt_make_contig(align_tail),
                                                       sbuf.data(), align_tail));
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        wait(sstatus);
        ssize += align_tail;
    }

    EXPECT_EQ(size_t(0), (ssize % dt_elem_size));

    std::vector<T> rbuf(ssize / dt_elem_size, 'r');
    size_t         roffset = 0;
    do {
        ucp::data_type_desc_t dt_desc(datatype, &rbuf[roffset / dt_elem_size],
                                      ssize - roffset);

        size_t length;
        void   *rreq = ucp_stream_recv_nb(receiver().ep(), dt_desc.buf(),
                                          dt_desc.count(), dt_desc.dt(),
                                          ucp_recv_cb, &length, 0);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq));
        if (UCS_PTR_IS_PTR(rreq)) {
            length = wait_stream_recv(rreq);
        }
        EXPECT_EQ(size_t(0), length % dt_elem_size);
        roffset += length;
    } while (roffset < ssize);

    EXPECT_EQ(roffset, ssize);
    const T     *check_ptr  = reinterpret_cast<const T *>(check_pattern.data());
    const size_t check_size = check_pattern.size() / dt_elem_size;
    EXPECT_EQ(std::vector<T>(check_ptr, check_ptr + check_size), rbuf);
}

template <typename T>
void test_ucp_stream::do_send_exp_recv_test(ucp_datatype_t datatype)
{
    const size_t dt_elem_size = UCP_DT_IS_CONTIG(datatype) ?
                                ucp_contig_dt_elem_size(datatype) : 1;
    const size_t msg_size = dt_elem_size * 1024 * 1024;
    const size_t n_msgs   = 10;

    std::vector<std::vector<T> > rbufs(n_msgs,
                                       std::vector<T>(msg_size / dt_elem_size, 'r'));
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
        size_t length = wait_stream_recv(rreqs[i]);
        EXPECT_EQ(size_t(0), length % dt_elem_size);
        rcount += length;
    }

    while (rcount < scount) {
        size_t           length = std::numeric_limits<size_t>::max();
        ucs_status_ptr_t rreq;
        rreq = ucp_stream_recv_nb(receiver().ep(), dt_rdescs[0].buf(),
                                  dt_rdescs[0].count(), dt_rdescs[0].dt(),
                                  ucp_recv_cb, &length, 0);
        if (UCS_PTR_IS_PTR(rreq)) {
            length = wait_stream_recv(rreq);
        }
        ASSERT_GT(length, 0ul);
        ASSERT_LE(length, msg_size);
        EXPECT_EQ(size_t(0), length % dt_elem_size);
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

void test_ucp_stream::do_send_recv_data_recv_test(ucp_datatype_t datatype)
{
    const size_t dt_elem_size = UCP_DT_IS_CONTIG(datatype) ?
                                ucp_contig_dt_elem_size(datatype) : 1;
    std::vector<char> sbuf(16 * 1024 * 1024, 's');
    size_t            ssize = 0; /* total send size */
    ucs_status_ptr_t  sstatus;
    std::vector<char> check_pattern;
    std::vector<char> rbuf;
    size_t            roffset = 0;
    ucs_status_ptr_t  rdata;
    size_t            length;

    size_t            send_i = dt_elem_size;
    size_t            recv_i = 0;
    do {
        if (send_i < sbuf.size()) {
            rbuf.resize(rbuf.size() + send_i, 'r');
            ucs::fill_random(sbuf, send_i);
            check_pattern.insert(check_pattern.end(), sbuf.begin(),
                                 sbuf.begin() + send_i);
            sstatus = stream_send_nb(ucp::data_type_desc_t(datatype, sbuf.data(),
                                                           send_i));
            EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
            wait(sstatus);
            ssize += send_i;
            send_i *= 2;
        }

        progress();

        if ((++recv_i % 2) || ((ssize - roffset) < dt_elem_size)) {
            rdata = ucp_stream_recv_data_nb(receiver().ep(), &length);
            if (UCS_PTR_STATUS(rdata) == UCS_OK) {
                continue;
            }

            memcpy(&rbuf[roffset], rdata, length);
            ucp_stream_data_release(receiver().ep(), rdata);
        } else {
            ucp::data_type_desc_t dt_desc(datatype, &rbuf[roffset], ssize - roffset);
            void *rreq = ucp_stream_recv_nb(receiver().ep(), dt_desc.buf(),
                                            dt_desc.count(), dt_desc.dt(),
                                            ucp_recv_cb, &length, 0);
            ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq));
            if (UCS_PTR_IS_PTR(rreq)) {
                length = wait_stream_recv(rreq);
            }
        }
        roffset += length;
    } while (roffset < ssize);

    EXPECT_EQ(roffset, ssize);
    EXPECT_EQ(check_pattern, rbuf);
}

UCS_TEST_P(test_ucp_stream, send_recv_data) {
    do_send_recv_data_test(DATATYPE);
}

UCS_TEST_P(test_ucp_stream, send_iov_recv_data) {
    do_send_recv_data_test(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream, send_recv_8) {
    do_send_recv_test<uint8_t>(ucp_dt_make_contig(sizeof(uint8_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_16) {
    do_send_recv_test<uint16_t>(ucp_dt_make_contig(sizeof(uint16_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_32) {
    do_send_recv_test<uint32_t>(ucp_dt_make_contig(sizeof(uint32_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_64) {
    do_send_recv_test<uint64_t>(ucp_dt_make_contig(sizeof(uint64_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_iov) {
    do_send_recv_test<uint8_t>(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_8) {
    do_send_exp_recv_test<uint8_t>(ucp_dt_make_contig(sizeof(uint8_t)));
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_16) {
    do_send_exp_recv_test<uint16_t>(ucp_dt_make_contig(sizeof(uint16_t)));
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_32) {
    do_send_exp_recv_test<uint32_t>(ucp_dt_make_contig(sizeof(uint32_t)));
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_64) {
    do_send_exp_recv_test<uint64_t>(ucp_dt_make_contig(sizeof(uint64_t)));
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_iov) {
    do_send_exp_recv_test<uint8_t>(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream, send_recv_data_recv_8) {
    do_send_recv_data_recv_test(ucp_dt_make_contig(sizeof(uint8_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_data_recv_16) {
    do_send_recv_data_recv_test(ucp_dt_make_contig(sizeof(uint16_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_data_recv_32) {
    do_send_recv_data_recv_test(ucp_dt_make_contig(sizeof(uint32_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_data_recv_64) {
    do_send_recv_data_recv_test(ucp_dt_make_contig(sizeof(uint64_t)));
}

UCS_TEST_P(test_ucp_stream, send_recv_data_recv_iov) {
    do_send_recv_data_recv_test(DATATYPE_IOV);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream)

class test_ucp_stream_many2one : public test_ucp_stream_base {
protected:
    struct request_wrapper_t {
        request_wrapper_t(void *request, ucp::data_type_desc_t *dt_desc)
            : m_req(request), m_dt_desc(dt_desc) {}

        void                  *m_req;
        ucp::data_type_desc_t *m_dt_desc;
    };

public:
    test_ucp_stream_many2one() : m_receiver_idx(3), m_nsenders(3) {
        m_recv_data.resize(m_nsenders);
    }

    static ucp_params_t get_ctx_params() {
        return test_ucp_stream::get_ctx_params();
    }

    virtual void init();
    static void ucp_send_cb(void *request, ucs_status_t status) {}
    static void ucp_recv_cb(void *request, ucs_status_t status, size_t length) {}

    void do_send_worker_poll_test(ucp_datatype_t dt);
    void do_send_recv_test(ucp_datatype_t dt);

protected:
    static void erase_completed_reqs(std::vector<request_wrapper_t> &reqs);
    ucs_status_ptr_t stream_send_nb(size_t sender_idx,
                                    const ucp::data_type_desc_t& dt_desc);
    size_t send_all_nb(ucp_datatype_t datatype, size_t n_iter,
                       std::vector<request_wrapper_t> &sreqs);
    size_t send_all(ucp_datatype_t datatype, size_t n_iter);
    void check_no_data();
    std::set<ucp_ep_h> check_no_data(entity &e);
    void check_recv_data(size_t n_iter);

    std::vector<std::string>        m_msgs;
    std::vector<std::vector<char> > m_recv_data;
    const size_t                    m_receiver_idx;
    const size_t                    m_nsenders;
};

void test_ucp_stream_many2one::init()
{
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

void test_ucp_stream_many2one::do_send_worker_poll_test(ucp_datatype_t dt)
{
    const size_t                   niter = 2018;
    std::vector<request_wrapper_t> sreqs;
    size_t                         total_len;

    total_len = send_all_nb(dt, niter, sreqs);

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

void test_ucp_stream_many2one::do_send_recv_test(ucp_datatype_t dt)
{
    const size_t                                       niter = 2018;
    std::vector<size_t>                                roffsets(m_nsenders, 0);
    std::vector<ucp::data_type_desc_t>                 dt_rdescs(m_nsenders);
    std::vector<std::pair<size_t, request_wrapper_t> > rreqs;
    std::vector<request_wrapper_t>                     sreqs;
    size_t                                             total_sdata;

    ASSERT_FALSE(m_msgs.empty());

    /* Do preposts */
    for (size_t i = 0; i < m_nsenders; ++i) {
        m_recv_data[i].resize(m_msgs[i].length() * niter + 1);
        ucp::data_type_desc_t &rdesc = dt_rdescs[i].make(dt,
                                                         &m_recv_data[i][roffsets[i]],
                                             m_recv_data[i].size());
        size_t length;
        void *rreq = ucp_stream_recv_nb(e(m_receiver_idx).ep(0, i),
                                        rdesc.buf(), rdesc.count(), rdesc.dt(),
                                        ucp_recv_cb, &length, 0);
        EXPECT_TRUE(UCS_PTR_IS_PTR(rreq));
        rreqs.push_back(std::make_pair(i, request_wrapper_t(rreq, &rdesc)));
    }

    total_sdata = send_all_nb(dt, niter, sreqs);

    /* Recv and progress all the rest of data */
    do {
        ssize_t count;
        /* wait rreqs */
        for (size_t i = 0; i < rreqs.size(); ++i) {
            roffsets[rreqs[i].first] += wait_stream_recv(rreqs[i].second.m_req);
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
                    rreqs.push_back(std::make_pair(sender_idx,
                                                   request_wrapper_t(rreq,
                                                                     &dt_desc)));
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

ucs_status_ptr_t
test_ucp_stream_many2one::stream_send_nb(size_t sender_idx,
                                         const ucp::data_type_desc_t& dt_desc)
{
    return ucp_stream_send_nb(m_entities.at(sender_idx).ep(), dt_desc.buf(),
                              dt_desc.count(), dt_desc.dt(), ucp_send_cb, 0);
}

size_t
test_ucp_stream_many2one::send_all_nb(ucp_datatype_t datatype, size_t n_iter,
                                      std::vector<request_wrapper_t> &sreqs)
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

            ucp::data_type_desc_t *dt_desc = new ucp::data_type_desc_t(datatype,
                                                                       buf,
                                                                       len);
            void *sreq = stream_send_nb(sender_idx, *dt_desc);
            total += len;
            if (UCS_PTR_IS_PTR(sreq)) {
                sreqs.push_back(request_wrapper_t(sreq, dt_desc));
            } else {
                EXPECT_FALSE(UCS_PTR_IS_ERR(sreq));
                delete dt_desc;
            }
        }
    }

    return total;
}

size_t
test_ucp_stream_many2one::send_all(ucp_datatype_t datatype, size_t n_iter)
{
    std::vector<request_wrapper_t> sreqs;
    size_t                         total;

    total = send_all_nb(datatype, n_iter, sreqs);
    while (!sreqs.empty()) {
        progress();
        erase_completed_reqs(sreqs);
    }

    return total;
}

void test_ucp_stream_many2one::check_no_data()
{
    std::set<ucp_ep_h> check;

    for (size_t i = 0; i <= m_receiver_idx; ++i) {
        std::set<ucp_ep_h> check_e = check_no_data(e(i));
        check.insert(check_e.begin(), check_e.end());
    }

    EXPECT_EQ(size_t(0), check.size());
}

std::set<ucp_ep_h> test_ucp_stream_many2one::check_no_data(entity &e)
{
    const size_t         max_eps = 10;
    ucp_stream_poll_ep_t poll_eps[max_eps];
    std::set<ucp_ep_h>   ret;
    std::list<ucp_ep_h>  check_list;

    while (progress());

    ssize_t count = ucp_stream_worker_poll(m_entities.at(m_receiver_idx).worker(),
                                           poll_eps, max_eps, 0);
    EXPECT_GE(count, ssize_t(0));

    for (ssize_t i = 0; i < count; ++i) {
        ret.insert(poll_eps[i].ep);
    }

    for (int i = 0; i < e.get_num_workers(); ++i) {
        for (int j = 0; j < e.get_num_eps(); ++j) {
            check_list.push_back(e.ep(i, j));
        }
    }

    std::list<ucp_ep_h>::const_iterator check_it = check_list.begin();
    while (check_it != check_list.end()) {
        EXPECT_EQ(ret.end(), ret.find(*check_it));
        ++check_it;
    }

    return ret;
}

void test_ucp_stream_many2one::check_recv_data(size_t n_iter)
{
    for (size_t i = 0; i < m_nsenders; ++i) {
        const std::string test = std::string("sender_") + ucs::to_string(i);
        const std::string str(&m_recv_data[i].front());
        size_t            next = 0;
        for (size_t j = 0; j < n_iter; ++j) {
            size_t match = str.find(test, next);
            EXPECT_NE(std::string::npos, match) << "failed on " << j
                                                << " iteration";
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
test_ucp_stream_many2one::erase_completed_reqs(std::vector<request_wrapper_t> &reqs)
{
    std::vector<request_wrapper_t>::iterator i = reqs.begin();

    while (i != reqs.end()) {
        ucs_status_t status = ucp_request_check_status(i->m_req);
        if (status != UCS_INPROGRESS) {
            EXPECT_EQ(UCS_OK, status);
            ucp_request_release(i->m_req);
            delete i->m_dt_desc;
            i = reqs.erase(i);
        } else {
            ++i;
        }
    }
}

UCS_TEST_P(test_ucp_stream_many2one, drop_data) {
    send_all(DATATYPE, 10);

    ASSERT_EQ(m_receiver_idx, m_nsenders);
    for (size_t i = 0; i <= m_receiver_idx; ++i) {
        flush_worker(e(i));
    }

    /* destroy 1 connection */
    entity::ep_destructor(m_entities.at(0).ep(),
                          &m_entities.at(0));
    entity::ep_destructor(m_entities.at(m_receiver_idx).ep(),
                          &m_entities.at(0));
    m_entities.at(0).revoke_ep();
    m_entities.at(m_receiver_idx).revoke_ep(0, 0);

    /* data from disconnected EP should be dropped */
    std::set<ucp_ep_h> others = check_no_data(m_entities.at(0));
    EXPECT_EQ(m_nsenders - 1, others.size());

    /* reconnect */
    m_entities.at(0).connect(&m_entities.at(m_receiver_idx), get_ep_params(), 0);
    ucp_ep_params_t recv_ep_param = get_ep_params();
    recv_ep_param.field_mask |= UCP_EP_PARAM_FIELD_USER_DATA;
    recv_ep_param.user_data   = (void *)uintptr_t(0xdeadbeef);
    e(m_receiver_idx).connect(&e(0), recv_ep_param, 0);

    /* send again */
    send_all(DATATYPE, 10);

    for (size_t i = 0; i <= m_receiver_idx; ++i) {
        flush_worker(e(i));
    }

    /* Need to poll out all incoming data from transport layer, see PR #2048 */
    while (progress() > 0);
}

UCS_TEST_P(test_ucp_stream_many2one, send_worker_poll) {
    do_send_worker_poll_test(DATATYPE);
}

UCS_TEST_P(test_ucp_stream_many2one, send_worker_poll_iov) {
    do_send_worker_poll_test(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream_many2one, send_recv_nb) {
    do_send_recv_test(DATATYPE);
}

UCS_TEST_P(test_ucp_stream_many2one, send_recv_nb_iov) {
    do_send_recv_test(DATATYPE_IOV);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream_many2one)
