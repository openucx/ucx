/**
* Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include "ucp_datatype.h"

#include <list>
#include <numeric>
#include <set>
#include <vector>

extern "C" {
#include <ucp/core/ucp_request.inl>
}


class test_ucp_stream_base : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_STREAM);
    }

    size_t wait_stream_recv(void *request);

protected:
    ucs_status_ptr_t stream_send_nb(const ucp::data_type_desc_t& dt_desc);
};

size_t test_ucp_stream_base::wait_stream_recv(void *request)
{
    ucs_time_t deadline = ucs::get_deadline();
    ucs_status_t status;
    size_t       length;
    do {
        progress();
        status = ucp_stream_recv_request_test(request, &length);
    } while ((status == UCS_INPROGRESS) && (ucs_get_time() < deadline));
    ASSERT_UCS_OK(status);
    ucp_request_free(request);

    return length;
}

ucs_status_ptr_t
test_ucp_stream_base::stream_send_nb(const ucp::data_type_desc_t& dt_desc)
{
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype     = dt_desc.dt();

    return ucp_stream_send_nbx(sender().ep(), dt_desc.buf(), dt_desc.count(),
                               &param);
}

class test_ucp_stream_onesided : public test_ucp_stream_base {
public:
    ucp_ep_params_t get_ep_params() {
        ucp_ep_params_t params = test_ucp_stream_base::get_ep_params();
        params.field_mask |= UCP_EP_PARAM_FIELD_FLAGS;
        params.flags      |= UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
        return params;
    }
};

UCS_TEST_P(test_ucp_stream_onesided, recv_not_connected_ep_cleanup) {
    receiver().connect(&sender(), get_ep_params());

    uint64_t recv_data = 0;
    size_t length;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_FLAGS;
    param.datatype     = ucp_dt_make_contig(sizeof(uint64_t));
    param.flags        = UCP_STREAM_RECV_FLAG_WAITALL;

    void *rreq = ucp_stream_recv_nbx(receiver().ep(), &recv_data, 1,
                                     &length, &param);
    EXPECT_TRUE(UCS_PTR_IS_PTR(rreq));
    EXPECT_EQ(UCS_INPROGRESS, ucp_request_check_status(rreq));
    disconnect(receiver());
    EXPECT_EQ(UCS_ERR_CANCELED, ucp_request_check_status(rreq));
    ucp_request_free(rreq);
}

UCS_TEST_P(test_ucp_stream_onesided, recv_connected_ep_cleanup) {
    skip_loopback();
    sender().connect(&receiver(), get_ep_params());
    receiver().connect(&sender(), get_ep_params());

    uint64_t send_data = ucs::rand();
    uint64_t recv_data = 0;
    ucp_datatype_t dt  = ucp_dt_make_contig(sizeof(uint64_t));

    ucp::data_type_desc_t send_dt_desc(dt, &send_data, sizeof(send_data));
    void *sreq = stream_send_nb(send_dt_desc);

    size_t recvd_length;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_FLAGS;
    param.datatype     = dt;
    param.flags        = UCP_STREAM_RECV_FLAG_WAITALL;

    void *rreq = ucp_stream_recv_nbx(receiver().ep(), &recv_data, 1,
                                     &recvd_length, &param);

    EXPECT_EQ(sizeof(send_data), wait_stream_recv(rreq));
    EXPECT_EQ(send_data, recv_data);
    request_wait(sreq);

    rreq = ucp_stream_recv_nbx(receiver().ep(), &recv_data, 1,
                               &recvd_length, &param);
    EXPECT_TRUE(UCS_PTR_IS_PTR(rreq));
    EXPECT_EQ(UCS_INPROGRESS, ucp_request_check_status(rreq));
    disconnect(sender());
    disconnect(receiver());
    EXPECT_EQ(UCS_ERR_CANCELED, ucp_request_check_status(rreq));
    ucp_request_free(rreq);
}

UCS_TEST_P(test_ucp_stream_onesided, send_recv_no_ep) {

    /* connect from sender side only and send */
    sender().connect(&receiver(), get_ep_params());
    uint64_t send_data = ucs::rand();
    ucp::data_type_desc_t dt_desc(ucp_dt_make_contig(sizeof(uint64_t)),
                                  &send_data, sizeof(send_data));
    void *sreq = stream_send_nb(dt_desc);
    request_wait(sreq);

    /* must not receive data before ep is created on receiver side */
    static const size_t max_eps = 10;
    ucp_stream_poll_ep_t poll_eps[max_eps];
    ssize_t count = ucp_stream_worker_poll(receiver().worker(), poll_eps,
                                           max_eps, 0);
    EXPECT_EQ(0l, count) << "ucp_stream_worker_poll returned ep too early";

    /* create receiver side ep */
    ucp_ep_params_t recv_ep_param = get_ep_params();
    recv_ep_param.field_mask |= UCP_EP_PARAM_FIELD_USER_DATA;
    recv_ep_param.user_data   = reinterpret_cast<void*>(static_cast<uintptr_t>(ucs::rand()));
    receiver().connect(&sender(), recv_ep_param);

    /* expect ep to be ready */
    ucs_time_t deadline = ucs_get_time() +
                          (ucs_time_from_sec(10.0) * ucs::test_time_multiplier());
    do {
        progress();
        count = ucp_stream_worker_poll(receiver().worker(), poll_eps, max_eps, 0);
    } while ((count == 0) && (ucs_get_time() < deadline));
    EXPECT_EQ(1l, count);
    EXPECT_EQ(recv_ep_param.user_data, poll_eps[0].user_data);
    EXPECT_EQ(receiver().ep(0), poll_eps[0].ep);

    /* expect data to be received */
    uint64_t recv_data = 0;
    size_t recv_length = 0;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype     = ucp_dt_make_contig(sizeof(uint64_t));
    void *rreq = ucp_stream_recv_nbx(receiver().ep(), &recv_data, 1,
                                     &recv_length, &param);
    ASSERT_UCS_PTR_OK(rreq);
    if (rreq != NULL) {
        recv_length = wait_stream_recv(rreq);
    }

    EXPECT_EQ(sizeof(uint64_t), recv_length);
    EXPECT_EQ(send_data, recv_data);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream_onesided)

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
    template <typename T, unsigned recv_flags>
    void do_send_recv_test(ucp_datatype_t datatype);
    template <typename T, unsigned recv_flags>
    void do_send_exp_recv_test(ucp_datatype_t datatype);
    void do_send_recv_data_recv_test(ucp_datatype_t datatype);

    /* for self-validation of generic datatype
     * NOTE: it's tested only with byte array data since it's recv completion
     *       granularity without UCP_RECV_FLAG_WAITALL flag */
    std::vector<uint8_t> context;
};

void test_ucp_stream::do_send_recv_data_test(ucp_datatype_t datatype)
{
    size_t            ssize = 0; /* total send size in bytes */
    std::vector<char> sbuf(16 * UCS_MBYTE, 's');
    std::vector<char> check_pattern;
    ucs_status_ptr_t  sstatus;

    /* send all msg sizes*/
    for (size_t i = 3; i < sbuf.size();
         i *= (2 * ucs::test_time_multiplier())) {
        if (UCP_DT_IS_GENERIC(datatype)) {
            for (size_t j = 0; j < i; ++j) {
                check_pattern.push_back(char(j));
            }
        } else {
            ucs::fill_random(sbuf, i);
            check_pattern.insert(check_pattern.end(), sbuf.begin(),
                                 sbuf.begin() + i);
        }
        ucp::data_type_desc_t dt_desc(datatype, sbuf.data(), i);
        sstatus = stream_send_nb(dt_desc);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        request_wait(sstatus);
        ssize += i;
    }

    std::vector<char> rbuf(ssize, 'r');
    size_t            roffset = 0;
    ucs_status_ptr_t  rdata;
    size_t length;
    do {
        progress();
        rdata = ucp_stream_recv_data_nb(receiver().ep(), &length);
        if (rdata == NULL) {
            continue;
        }

        memcpy(&rbuf[roffset], rdata, length);
        roffset += length;
        ucp_stream_data_release(receiver().ep(), rdata);
    } while (roffset < ssize);

    EXPECT_EQ(roffset, ssize);
    EXPECT_EQ(check_pattern, rbuf);
}

template <typename T, unsigned recv_flags>
void test_ucp_stream::do_send_recv_test(ucp_datatype_t datatype)
{
    const size_t      dt_elem_size    = UCP_DT_IS_CONTIG(datatype) ?
                                        ucp_contig_dt_elem_size(datatype) : 1;
    size_t            ssize           = 0; /* total send size */
    size_t            iter_multiplier = RUNNING_ON_VALGRIND ? 10 : 2;
    std::vector<char> sbuf(16 * UCS_MBYTE / ucs::test_time_multiplier(), 's');
    ucs_status_ptr_t  sstatus;
    std::vector<char> check_pattern;

    /* send all msg sizes in bytes*/
    for (size_t i = 3; i < sbuf.size(); i *= iter_multiplier) {
        ucp_datatype_t dt;
        if (UCP_DT_IS_GENERIC(datatype)) {
            dt = datatype;
            for (size_t j = 0; j < i; ++j) {
                context.push_back(uint8_t(j));
            }
        } else {
            dt = DATATYPE;
            ucs::fill_random(sbuf, i);
            check_pattern.insert(check_pattern.end(), sbuf.begin(),
                                 sbuf.begin() + i);
        }
        ucp::data_type_desc_t dt_desc(dt, sbuf.data(), i);
        sstatus = stream_send_nb(dt_desc);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        request_wait(sstatus);
        ssize += i;
    }

    size_t align_tail = UCP_DT_IS_GENERIC(datatype) ? 0 :
                        (dt_elem_size - ssize % dt_elem_size);
    if (align_tail != 0) {
        ucs::fill_random(sbuf, align_tail);
        check_pattern.insert(check_pattern.end(), sbuf.begin(), sbuf.begin() + align_tail);
        ucp::data_type_desc_t dt_desc(ucp_dt_make_contig(align_tail),
                                      sbuf.data(), align_tail);
        sstatus = stream_send_nb(dt_desc);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        request_wait(sstatus);
        ssize += align_tail;
    }

    EXPECT_EQ(size_t(0), (ssize % dt_elem_size));

    std::vector<T> rbuf(ssize / dt_elem_size, 'r');
    size_t         roffset = 0;
    size_t         counter = 0;
    do {
        ucp::data_type_desc_t dt_desc(datatype, &rbuf[roffset / dt_elem_size],
                                      ssize - roffset);

        size_t length;
        ucp_request_param_t param;

        param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                             UCP_OP_ATTR_FIELD_FLAGS;
        param.datatype     = dt_desc.dt();
        param.flags        = recv_flags;

        void *rreq = ucp_stream_recv_nbx(receiver().ep(), dt_desc.buf(),
                                         dt_desc.count(), &length, &param);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq));
        if (UCS_PTR_IS_PTR(rreq)) {
            length = wait_stream_recv(rreq);
        }
        EXPECT_EQ(size_t(0), length % dt_elem_size);
        roffset += length;
        counter++;
    } while (roffset < ssize);

    /* waitall flag requires completion by single request */
    if (recv_flags & UCP_STREAM_RECV_FLAG_WAITALL) {
        EXPECT_EQ(size_t(1), counter);
    }

    EXPECT_EQ(roffset, ssize);
    if (!UCP_DT_IS_GENERIC(datatype)) {
        const T     *check_ptr  = reinterpret_cast<const T *>(check_pattern.data());
        const size_t check_size = check_pattern.size() / dt_elem_size;
        EXPECT_EQ(std::vector<T>(check_ptr, check_ptr + check_size), rbuf);
    }
}

template <typename T, unsigned recv_flags>
void test_ucp_stream::do_send_exp_recv_test(ucp_datatype_t datatype)
{
    const size_t dt_elem_size = UCP_DT_IS_CONTIG(datatype) ?
                                ucp_contig_dt_elem_size(datatype) : 1;
    const size_t msg_size     = dt_elem_size *
                                /* message size must be a multiple of
                                 * dt_elem_size */
                                (UCS_MBYTE / ucs::test_time_multiplier());
    const size_t n_msgs       = ucs_max(2, 10 / ucs::test_time_multiplier());

    std::vector<std::vector<T> > rbufs(n_msgs,
                                       std::vector<T>(msg_size / dt_elem_size, 'r'));
    std::vector<ucp::data_type_desc_t> dt_rdescs(n_msgs);
    std::vector<void *> rreqs;

    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_FLAGS;
    param.flags        = recv_flags;

    /* post recvs */
    for (size_t i = 0; i < n_msgs; ++i) {
        ucp::data_type_desc_t &rdesc = dt_rdescs[i].make(datatype, &rbufs[i][0],
                                                         msg_size);
        size_t length;

        param.datatype = rdesc.dt();

        void *rreq = ucp_stream_recv_nbx(receiver().ep(), rdesc.buf(),
                                         rdesc.count(), &length, &param);
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
        request_wait(sreq);
        scount += sbuf.size();
    }

    size_t rcount = 0;
    for (size_t i = 0; i < rreqs.size(); ++i) {
        size_t length = wait_stream_recv(rreqs[i]);
        EXPECT_EQ(size_t(0), length % dt_elem_size);
        rcount += length;
    }

    size_t counter = 0;
    param.flags    = 0;
    param.datatype = dt_rdescs[0].dt();
    while (rcount < scount) {
        size_t           length = std::numeric_limits<size_t>::max();
        ucs_status_ptr_t rreq;

        rreq = ucp_stream_recv_nbx(receiver().ep(), dt_rdescs[0].buf(),
                                   dt_rdescs[0].count(), &length, &param);
        if (UCS_PTR_IS_PTR(rreq)) {
            length = wait_stream_recv(rreq);
        }
        ASSERT_GT(length, 0ul);
        ASSERT_LE(length, msg_size);
        EXPECT_EQ(size_t(0), length % dt_elem_size);
        rcount += length;
        counter++;
    }
    EXPECT_EQ(scount, rcount);

    /* waitall flag requires completion by single request */
    if (recv_flags & UCP_STREAM_RECV_FLAG_WAITALL) {
        EXPECT_EQ(size_t(0), counter);
    }

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
    size_t            ssize   = 0; /* total send size */
    size_t            roffset = 0;
    size_t            send_i  = dt_elem_size;
    size_t            recv_i  = 0;
    std::vector<char> sbuf(16 * UCS_MBYTE, 's');
    ucs_status_ptr_t  sstatus;
    std::vector<char> check_pattern;
    std::vector<char> rbuf;
    ucs_status_ptr_t  rdata;
    size_t            length;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;

    do {
        if (send_i < sbuf.size()) {
            rbuf.resize(rbuf.size() + send_i, 'r');
            ucs::fill_random(sbuf, send_i);
            check_pattern.insert(check_pattern.end(), sbuf.begin(),
                                 sbuf.begin() + send_i);
            ucp::data_type_desc_t dt_desc(datatype, sbuf.data(), send_i);
            sstatus = stream_send_nb(dt_desc);
            EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
            request_wait(sstatus);
            ssize += send_i;
            send_i *= 2;
        }

        progress();

        if ((++recv_i % 2) || ((ssize - roffset) < dt_elem_size)) {
            rdata = ucp_stream_recv_data_nb(receiver().ep(), &length);
            if (rdata == NULL) {
                continue;
            }

            memcpy(&rbuf[roffset], rdata, length);
            ucp_stream_data_release(receiver().ep(), rdata);
        } else {
            ucp::data_type_desc_t dt_desc(datatype, &rbuf[roffset], ssize - roffset);
            param.datatype = dt_desc.dt();
            void *rreq = ucp_stream_recv_nbx(receiver().ep(), dt_desc.buf(),
                                             dt_desc.count(), &length, &param);
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

UCS_TEST_P(test_ucp_stream, send_generic_recv_data) {
    ucp_datatype_t dt;
    ucs_status_t status;

    status = ucp_dt_create_generic(&ucp::test_dt_uint8_ops, NULL, &dt);
    ASSERT_UCS_OK(status);
    do_send_recv_data_test(dt);
    ucp_dt_destroy(dt);
}

UCS_TEST_P(test_ucp_stream, send_recv_8) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint8_t));

    do_send_recv_test<uint8_t, 0>(datatype);
    do_send_recv_test<uint8_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_recv_16) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint16_t));

    do_send_recv_test<uint16_t, 0>(datatype);
    do_send_recv_test<uint16_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_recv_32) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint32_t));

    do_send_recv_test<uint32_t, 0>(datatype);
    do_send_recv_test<uint32_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_recv_64) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint64_t));

    do_send_recv_test<uint64_t, 0>(datatype);
    do_send_recv_test<uint64_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_recv_iov) {
    do_send_recv_test<uint8_t, 0>(DATATYPE_IOV);
    do_send_recv_test<uint8_t, UCP_STREAM_RECV_FLAG_WAITALL>(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream, send_recv_generic) {
    ucp_datatype_t dt;
    ucs_status_t status;

    status = ucp_dt_create_generic(&ucp::test_dt_uint8_ops, &context, &dt);
    ASSERT_UCS_OK(status);
    do_send_recv_test<uint8_t, UCP_STREAM_RECV_FLAG_WAITALL>(dt);
    ucp_dt_destroy(dt);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_8) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint8_t));

    do_send_exp_recv_test<uint8_t, 0>(datatype);
    do_send_exp_recv_test<uint8_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_16) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint16_t));

    do_send_exp_recv_test<uint16_t, 0>(datatype);
    do_send_exp_recv_test<uint16_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_32) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint32_t));

    do_send_exp_recv_test<uint32_t, 0>(datatype);
    do_send_exp_recv_test<uint32_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_64) {
    ucp_datatype_t datatype = ucp_dt_make_contig(sizeof(uint64_t));
    const uct_md_attr_t *md_attr = ucp_ep_md_attr(sender().ep(), 0);

    if (has_transport("shm") && (md_attr->cap.max_alloc < UCS_GBYTE)) {
        UCS_TEST_SKIP_R("Not enough shared memory");
    }

    do_send_exp_recv_test<uint64_t, 0>(datatype);
    do_send_exp_recv_test<uint64_t, UCP_STREAM_RECV_FLAG_WAITALL>(datatype);
}

UCS_TEST_P(test_ucp_stream, send_exp_recv_iov) {
    do_send_exp_recv_test<uint8_t, 0>(DATATYPE_IOV);
    do_send_exp_recv_test<uint8_t, UCP_STREAM_RECV_FLAG_WAITALL>(DATATYPE_IOV);
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

UCS_TEST_P(test_ucp_stream, send_zero_ending_iov_recv_data) {
    const size_t min_size         = UCS_KBYTE;
    const size_t max_size         = min_size * 64 / ucs::test_time_multiplier();
    const size_t step_size        = RUNNING_ON_VALGRIND ? 111 : 1;
    const size_t iov_num          = 8; /* must be divisible by 4 without a
                                        * remainder, caught on mlx5 based TLs
                                        * where max_iov = 3 for zcopy multi
                                        * protocol, where every posting includes:
                                        * 1 header + 2 nonempty IOVs */
    const size_t iov_num_nonempty = iov_num / 2;

    std::vector<uint8_t> buf(max_size * 2);
    ucs::fill_random(buf, buf.size());
    std::vector<ucp_dt_iov_t> v(iov_num);

    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype     = DATATYPE_IOV;

    for (size_t size = min_size; size < max_size; size += step_size) {
        size_t slen = 0;
        for (size_t j = 0; j < iov_num; ++j) {
            if ((j % 2) == 0) {
                uint8_t *ptr = buf.data();
                v[j].buffer = &(ptr[j * size / iov_num_nonempty]);
                v[j].length = size / iov_num_nonempty;
                slen       += v[j].length;
            } else {
                v[j].buffer = NULL;
                v[j].length = 0;
            }
        }

        void *sreq = ucp_stream_send_nbx(sender().ep(), &v[0], iov_num, &param);

        size_t rlen = 0;
        while (rlen < slen) {
            progress();
            size_t length;
            void *rdata = ucp_stream_recv_data_nb(receiver().ep(), &length);
            EXPECT_FALSE(UCS_PTR_IS_ERR(rdata));
            if (rdata != NULL) {
                rlen += length;
                ucp_stream_data_release(receiver().ep(), rdata);
            }
        }
        request_wait(sreq);
    }
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

    virtual void init();

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
    void check_recv_data(size_t n_iter, ucp_datatype_t dt);

    std::vector<std::string>        m_msgs;
    std::vector<std::vector<char> > m_recv_data;
    const size_t                    m_receiver_idx;
    const size_t                    m_nsenders;
};

void test_ucp_stream_many2one::init()
{
    if (is_self()) {
        UCS_TEST_SKIP_R("self");
    }

    /* Skip entities creation */
    test_base::init();

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
    check_recv_data(niter, dt);
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
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;

    /* Do preposts */
    for (size_t i = 0; i < m_nsenders; ++i) {
        m_recv_data[i].resize(m_msgs[i].length() * niter + 1);
        ucp::data_type_desc_t &rdesc = dt_rdescs[i].make(dt,
                                                         &m_recv_data[i][roffsets[i]],
                                                         m_recv_data[i].size());
        size_t length;
        param.datatype = rdesc.dt();
        void *rreq = ucp_stream_recv_nbx(e(m_receiver_idx).ep(0, i),
                                         rdesc.buf(), rdesc.count(), &length,
                                         &param);
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
        progress();

        const size_t max_eps = 10;
        ucp_stream_poll_ep_t poll_eps[max_eps];
        count = ucp_stream_worker_poll(e(m_receiver_idx).worker(),
                                       poll_eps, max_eps, 0);
        EXPECT_LE(0, count);
        EXPECT_LE(size_t(count), m_nsenders);

        for (ssize_t i = 0; i < count; ++i) {
            bool again = true;
            while (again) {
                size_t sender_idx = uintptr_t(poll_eps[i].user_data);
                size_t &roffset   = roffsets[sender_idx];
                ucp::data_type_desc_t &dt_desc =
                    dt_rdescs[sender_idx].forward_to(roffset);
                size_t length;
                param.datatype = dt_desc.dt();
                void *rreq = ucp_stream_recv_nbx(poll_eps[i].ep, dt_desc.buf(),
                                                 dt_desc.count(), &length,
                                                 &param);
                EXPECT_FALSE(UCS_PTR_IS_ERR(rreq));
                if (rreq == NULL) {
                    EXPECT_LT(size_t(0), length);
                    roffset += length;
                    if (ssize_t(length) < dt_desc.buf_length()) {
                        continue; /* Need to drain the EP */
                    }
                } else {
                    rreqs.push_back(std::make_pair(sender_idx,
                                                   request_wrapper_t(rreq,
                                                                     &dt_desc)));
                }
                again = false;
            }
        }

        erase_completed_reqs(sreqs);
    } while (!rreqs.empty() || !sreqs.empty() ||
             (total_sdata > std::accumulate(roffsets.begin(),
                                            roffsets.end(), 0ul)));

    EXPECT_EQ(total_sdata, std::accumulate(roffsets.begin(),
                                           roffsets.end(), 0ul));
    check_no_data();
    check_recv_data(niter, dt);
}

ucs_status_ptr_t
test_ucp_stream_many2one::stream_send_nb(size_t sender_idx,
                                         const ucp::data_type_desc_t& dt_desc)
{
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype     = dt_desc.dt();

    return ucp_stream_send_nbx(m_entities.at(sender_idx).ep(), dt_desc.buf(),
                               dt_desc.count(), &param);
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

void test_ucp_stream_many2one::check_recv_data(size_t n_iter, ucp_datatype_t dt)
{
    for (size_t i = 0; i < m_nsenders; ++i) {
        std::string test = std::string("sender_") + ucs::to_string(i);
        const std::string str(&m_recv_data[i].front());
        if (UCP_DT_IS_GENERIC(dt)) {
            std::vector<char> test_gen;
            for (size_t j = 0; j < test.length(); ++j) {
                test_gen.push_back(char(j));
            }
            test_gen.push_back('\0');
            test = std::string(test_gen.data());
        }

        std::string::size_type next = 0;
        for (size_t j = 0; j < n_iter; ++j) {
            std::string::size_type match = str.find(test, next);
            EXPECT_NE(std::string::npos, match) << "failed on sender " << i
                                                << " iteration " << j;
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
            ucp_request_free(i->m_req);
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

    /* wait for 1-st byte on the last EP to be sure the network packets have
       been arrived */
    uint8_t check;
    size_t  check_length;
    ucp_ep_h last_ep = m_entities.at(m_receiver_idx).ep(0, m_nsenders - 1);
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    param.datatype     = DATATYPE;
    void *check_req    = ucp_stream_recv_nbx(last_ep, &check, 1, &check_length,
                                             &param);
    EXPECT_FALSE(UCS_PTR_IS_ERR(check_req));
    if (UCS_PTR_IS_PTR(check_req)) {
        wait_stream_recv(check_req);
    }

    /* data from disconnected EP should be dropped */
    std::set<ucp_ep_h> others = check_no_data(m_entities.at(0));
    /* since ordering between EPs is not guaranteed, some data may be still in
     * the network or buffered by transport */
    EXPECT_LE(others.size(), m_nsenders - 1);

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

UCS_TEST_P(test_ucp_stream_many2one, send_worker_poll_generic) {
    ucp_datatype_t dt;
    ucs_status_t status;

    status = ucp_dt_create_generic(&ucp::test_dt_uint8_ops, NULL, &dt);
    ASSERT_UCS_OK(status);
    do_send_worker_poll_test(dt);
    ucp_dt_destroy(dt);
}

UCS_TEST_P(test_ucp_stream_many2one, send_recv_nb) {
    do_send_recv_test(DATATYPE);
}

UCS_TEST_P(test_ucp_stream_many2one, send_recv_nb_iov) {
    do_send_recv_test(DATATYPE_IOV);
}

UCS_TEST_P(test_ucp_stream_many2one, send_recv_nb_generic) {
    ucp_datatype_t dt;
    ucs_status_t status;

    status = ucp_dt_create_generic(&ucp::test_dt_uint8_ops, NULL, &dt);
    ASSERT_UCS_OK(status);
    do_send_recv_test(dt);
    ucp_dt_destroy(dt);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_stream_many2one)
