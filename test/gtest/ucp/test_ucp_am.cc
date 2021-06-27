/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Los Alamos National Security, LLC. 2018. ALL RIGHTS RESERVED.
 *
 */
#include <list>
#include <numeric>
#include <set>
#include <vector>
#include <math.h>

#include <common/test.h>
#include <common/mem_buffer.h>

#include "ucp_datatype.h"
#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_ep.inl>
}

#define NUM_MESSAGES 17

#define UCP_REALLOC_ID 1000
#define UCP_SEND_ID 0
#define UCP_REPLY_ID 1
#define UCP_RELEASE 1

class test_ucp_am_base : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_AM);
    }

    virtual void init() {
        modify_config("MAX_EAGER_LANES", "2");

        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());
    }
};

class test_ucp_am : public test_ucp_am_base {
public:
    int sent_ams;
    int replies;
    int recv_ams;
    void *reply;
    void *for_release[NUM_MESSAGES];
    int release;

    static ucs_status_t ucp_process_am_cb(void *arg, void *data,
                                          size_t length,
                                          ucp_ep_h reply_ep,
                                          unsigned flags);

    static ucs_status_t ucp_process_reply_cb(void *arg, void *data,
                                             size_t length,
                                             ucp_ep_h reply_ep,
                                             unsigned flags);

    ucs_status_t am_handler(test_ucp_am *me, void *data,
                            size_t  length, unsigned flags);

protected:
    void do_set_am_handler_realloc_test();
    void do_send_process_data_test(int test_release, uint16_t am_id,
                                   int send_reply);
    void do_send_process_data_iov_test(size_t size);
    void set_handlers(uint16_t am_id);
    void set_reply_handlers();
};

ucs_status_t test_ucp_am::ucp_process_reply_cb(void *arg, void *data,
                                               size_t length,
                                               ucp_ep_h reply_ep,
                                               unsigned flags)
{
    test_ucp_am *self = reinterpret_cast<test_ucp_am*>(arg);
    self->replies++;
    return UCS_OK;
}

ucs_status_t test_ucp_am::ucp_process_am_cb(void *arg, void *data,
                                            size_t length,
                                            ucp_ep_h reply_ep,
                                            unsigned flags)
{
    test_ucp_am *self = reinterpret_cast<test_ucp_am*>(arg);

    if (reply_ep) {
        self->reply = ucp_am_send_nb(reply_ep, UCP_REPLY_ID, NULL, 1,
                                     ucp_dt_make_contig(0),
                                     (ucp_send_callback_t) ucs_empty_function,
                                     0);
        EXPECT_FALSE(UCS_PTR_IS_ERR(self->reply));
    }

    return self->am_handler(self, data, length, flags);
}

ucs_status_t test_ucp_am::am_handler(test_ucp_am *me, void *data,
                                     size_t length, unsigned flags)
{
    ucs_status_t status;
    std::vector<char> cmp(length, (char)length);
    std::vector<char> databuf(length, 'r');

    memcpy(&databuf[0], data, length);

    EXPECT_EQ(cmp, databuf);

    bool has_desc = flags & UCP_CB_PARAM_FLAG_DATA;
    if (me->release) {
        me->for_release[me->recv_ams] = has_desc ? data : NULL;
        status                        = has_desc ? UCS_INPROGRESS : UCS_OK;
    } else {
        status                        = UCS_OK;
    }

    me->recv_ams++;
    return status;
}


void test_ucp_am::set_reply_handlers()
{
    ucp_worker_set_am_handler(sender().worker(), UCP_REPLY_ID,
                              ucp_process_reply_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
    ucp_worker_set_am_handler(receiver().worker(), UCP_REPLY_ID,
                              ucp_process_reply_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
}

void test_ucp_am::set_handlers(uint16_t am_id)
{
    ucp_worker_set_am_handler(sender().worker(), am_id,
                              ucp_process_am_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
    ucp_worker_set_am_handler(receiver().worker(), am_id,
                              ucp_process_am_cb, this,
                              UCP_AM_FLAG_WHOLE_MSG);
}

void test_ucp_am::do_send_process_data_test(int test_release, uint16_t am_id,
                                            int send_reply)
{
    size_t buf_size          = pow(2, NUM_MESSAGES - 2);
    ucs_status_ptr_t sstatus = NULL;
    std::vector<char> buf(buf_size);

    recv_ams      = 0;
    sent_ams      = 0;
    replies       = 0;
    this->release = test_release;

    for (size_t i = 0; i < buf_size + 1; i = i ? (i * 2) : 1) {
        for (size_t j = 0; j < i; j++) {
            buf[j] = i;
        }

        reply   = NULL;
        sstatus = ucp_am_send_nb(sender().ep(), am_id,
                                 buf.data(), 1, ucp_dt_make_contig(i),
                                 (ucp_send_callback_t) ucs_empty_function,
                                 send_reply);

        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        request_wait(sstatus);
        sent_ams++;

        if (send_reply) {
            while (sent_ams != replies) {
                progress();
            }

            if (reply != NULL) {
                ucp_request_release(reply);
            }
        }
    }

    while (sent_ams != recv_ams) {
        progress();
    }

    if (send_reply) {
        while (sent_ams != replies) {
            progress();
        }
    }

    if (test_release) {
        for(int i = 0; i < recv_ams; i++) {
            if (for_release[i] != NULL) {
                ucp_am_data_release(receiver().worker(), for_release[i]);
            }
        }
    }
}

void test_ucp_am::do_send_process_data_iov_test(size_t size)
{
    ucs_status_ptr_t sstatus;
    size_t index;
    size_t i;

    recv_ams = 0;
    sent_ams = 0;
    release  = 0;

    const size_t iovcnt = 2;
    std::vector<char> sendbuf(size * iovcnt, 0);

    ucs::fill_random(sendbuf);

    set_handlers(0);

    for (i = 1; i < size; i *= 2) {
        for (size_t iov_it = 0; iov_it < iovcnt; iov_it++) {
            for (index = 0; index < i; index++) {
                sendbuf[(iov_it * i) + index] = i * 2;
            }
        }

        ucp::data_type_desc_t send_dt_desc(DATATYPE_IOV, sendbuf.data(),
                                           i * iovcnt, iovcnt);

        sstatus = ucp_am_send_nb(sender().ep(), 0,
                                 send_dt_desc.buf(), iovcnt, DATATYPE_IOV,
                                 (ucp_send_callback_t) ucs_empty_function, 0);
        request_wait(sstatus);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sstatus));
        sent_ams++;
    }

    while (sent_ams != recv_ams) {
        progress();
    }
}

void test_ucp_am::do_set_am_handler_realloc_test()
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(0, UCP_SEND_ID, 0);

    set_handlers(UCP_REALLOC_ID);
    do_send_process_data_test(0, UCP_REALLOC_ID, 0);

    set_handlers(UCP_SEND_ID + 1);
    do_send_process_data_test(0, UCP_SEND_ID + 1, 0);
}

UCS_TEST_P(test_ucp_am, send_process_am)
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(0, UCP_SEND_ID, 0);

    set_reply_handlers();
    do_send_process_data_test(0, UCP_SEND_ID, UCP_AM_SEND_REPLY);
}

UCS_TEST_P(test_ucp_am, send_process_am_rndv, "RNDV_THRESH=1")
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(0, UCP_SEND_ID, 0);

    set_reply_handlers();
    do_send_process_data_test(0, UCP_SEND_ID, UCP_AM_SEND_REPLY);
}

UCS_TEST_P(test_ucp_am, send_process_am_release)
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(UCP_RELEASE, 0, 0);
}

UCS_TEST_P(test_ucp_am, send_process_iov_am)
{
    ucs::detail::message_stream ms("INFO");

    for (unsigned i = 1; i <= 7; ++i) {
        size_t max = (long)pow(10.0, i);
        long count = ucs_max((long)(5000.0 / sqrt(max) /
                                    ucs::test_time_multiplier()), 3);
        ms << count << "x10^" << i << " " << std::flush;
        for (long j = 0; j < count; ++j) {
            size_t size = ucs::rand() % max + 1;
            do_send_process_data_iov_test(size);
        }
    }
}

UCS_TEST_P(test_ucp_am, set_am_handler_realloc)
{
    do_set_am_handler_realloc_test();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am)


class test_ucp_am_nbx : public test_ucp_am_base {
public:
    static const uint64_t SEED = 0x1111111111111111lu;

    test_ucp_am_nbx()
    {
        m_dt          = ucp_dt_make_contig(1);
        m_am_received = false;
        m_rx_dt       = ucp_dt_make_contig(1);
        m_rx_memtype  = UCS_MEMORY_TYPE_HOST;
        m_rx_buf      = NULL;
    }

protected:
    size_t max_am_hdr()
    {
        ucp_worker_attr_t attr;
        attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;

        ASSERT_UCS_OK(ucp_worker_query(sender().worker(), &attr));
        return attr.max_am_header;
    }

    virtual unsigned get_send_flag()
    {
        return 0;
    }

    ucp_datatype_t make_dt(int dt)
    {
        if (dt == UCP_DATATYPE_CONTIG) {
           return ucp_dt_make_contig(1);
        } else if (dt == UCP_DATATYPE_IOV) {
           return ucp_dt_make_iov();
        } else {
            ucs_assert(UCP_DATATYPE_GENERIC == dt);
            ucp_datatype_t ucp_dt;
            ASSERT_UCS_OK(ucp_dt_create_generic(&ucp::test_dt_copy_ops, NULL,
                                                &ucp_dt));
            return ucp_dt;
        }
    }

    void destroy_dt(ucp_datatype_t dt)
    {
        if (UCP_DT_IS_GENERIC(dt)) {
            ucp_dt_destroy(dt);
        }
    }

    void set_am_data_handler(entity &e, uint16_t am_id, ucp_am_recv_callback_t cb,
                             void *arg, unsigned flags = 0)
    {
        ucp_am_handler_param_t param;

        /* Initialize Active Message data handler */
        param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                           UCP_AM_HANDLER_PARAM_FIELD_CB |
                           UCP_AM_HANDLER_PARAM_FIELD_ARG;
        param.id         = am_id;
        param.cb         = cb;
        param.arg        = arg;

        if (flags != 0) {
            param.field_mask |= UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
            param.flags       = flags;
        }

        ASSERT_UCS_OK(ucp_worker_set_am_recv_handler(e.worker(), &param));
    }

    void check_header(const void *header, size_t header_length)
    {
        std::string check_pattern((char*)header, header_length);
        EXPECT_EQ(check_pattern, m_hdr);
    }

    ucs_status_ptr_t send_am(const ucp::data_type_desc_t& dt_desc,
                             unsigned flags = 0, const void *hdr = NULL,
                             unsigned hdr_length = 0)
    {
        ucp_request_param_t param;
        param.op_attr_mask      = UCP_OP_ATTR_FIELD_DATATYPE;
        param.datatype          = dt_desc.dt();

        if (flags != 0) {
            param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
            param.flags         = flags;
        }

        ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID,
                                                hdr, hdr_length, dt_desc.buf(),
                                                dt_desc.count(), &param);
        return sptr;
    }

    void test_am_send_recv(size_t size, size_t header_size = 0ul,
                           unsigned flags = 0,
                           ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST,
                           unsigned data_cb_flags = 0)
    {
        mem_buffer sbuf(size, mem_type);
        mem_buffer::pattern_fill(sbuf.ptr(), size, SEED, mem_type);
        m_hdr.resize(header_size);
        ucs::fill_random(m_hdr);
        m_am_received = false;

        set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_cb, this,
                            data_cb_flags);

        ucp::data_type_desc_t sdt_desc(m_dt, sbuf.ptr(), size);

        ucs_status_ptr_t sptr = send_am(sdt_desc, get_send_flag() | flags,
                                        m_hdr.data(), m_hdr.size());

        wait_for_flag(&m_am_received);
        request_wait(sptr);
        EXPECT_TRUE(m_am_received);
    }

    void test_am(size_t size, unsigned flags = 0)
    {
        size_t small_hdr_size = 8;

        test_am_send_recv(size, 0, flags);
        test_am_send_recv(size, small_hdr_size, flags);

        if (max_am_hdr() > small_hdr_size) {
            test_am_send_recv(size, max_am_hdr(), flags);
        }
    }

    void test_short_thresh(size_t max_short)
    {
        ucp_ep_config_t *ep_cfg = ucp_ep_config(sender().ep());

        EXPECT_LE(max_short, ep_cfg->rndv.am_thresh.remote);
        EXPECT_LE(max_short, ep_cfg->rndv.am_thresh.local);
        EXPECT_LE(max_short, ep_cfg->rndv.rma_thresh.remote);
        EXPECT_LE(max_short, ep_cfg->rndv.rma_thresh.local);
    }

    virtual ucs_status_t am_data_handler(const void *header,
                                         size_t header_length,
                                         void *data, size_t length,
                                         const ucp_am_recv_param_t *rx_param)
    {
        ucs_status_t status;

        EXPECT_FALSE(m_am_received);

        check_header(header, header_length);

        bool has_reply_ep = get_send_flag();

        EXPECT_EQ(has_reply_ep, rx_param->recv_attr &
                                UCP_AM_RECV_ATTR_FIELD_REPLY_EP);
        EXPECT_EQ(has_reply_ep, rx_param->reply_ep != NULL);

        if (!(rx_param->recv_attr &
              (UCP_AM_RECV_ATTR_FLAG_RNDV | UCP_AM_RECV_ATTR_FLAG_DATA))) {
            mem_buffer::pattern_check(data, length, SEED);
            m_am_received = true;
            return UCS_OK;
        }

        m_rx_buf = mem_buffer::allocate(length, m_rx_memtype);
        mem_buffer::pattern_fill(m_rx_buf, length, 0ul, m_rx_memtype);

        m_rx_dt_desc.make(m_rx_dt, m_rx_buf, length);

        uint32_t imm_compl_flag = UCP_OP_ATTR_FLAG_NO_IMM_CMPL *
                                  (ucs::rand() % 2);
        size_t rx_length = SIZE_MAX;
        ucp_request_param_t params;
        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA |
                              UCP_OP_ATTR_FIELD_DATATYPE |
                              UCP_OP_ATTR_FIELD_RECV_INFO |
                              imm_compl_flag;
        params.datatype     = m_rx_dt_desc.dt();
        params.cb.recv_am   = am_data_recv_cb;
        params.user_data    = this;
        params.recv_info.length = &rx_length;
        ucs_status_ptr_t sp = ucp_am_recv_data_nbx(receiver().worker(),
                                                   data, m_rx_dt_desc.buf(),
                                                   m_rx_dt_desc.count(),
                                                   &params);
        //ucs_warn("imm_compl %d, sp %p, rx len %zu", imm_compl_flag, sp, rx_length);
        if (UCS_PTR_IS_PTR(sp)) {
            ucp_request_release(sp);
            status = UCS_INPROGRESS;
        } else {
            EXPECT_EQ(NULL, sp);
            EXPECT_EQ(rx_length, length);
            am_recv_check_data(rx_length);
            status = UCS_OK;
        }

        return status;
    }

    void am_recv_check_data(size_t length)
    {
        ASSERT_FALSE(m_am_received);
        m_am_received = true;
        mem_buffer::pattern_check(m_rx_buf, length, SEED, m_rx_memtype);
        mem_buffer::release(m_rx_buf, m_rx_memtype);
    }

    static ucs_status_t am_data_cb(void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param)
    {
        test_ucp_am_nbx *self = reinterpret_cast<test_ucp_am_nbx*>(arg);
        return self->am_data_handler(header, header_length, data, length, param);
    }

    static ucs_status_t am_rx_check_cb(void *arg, const void *header,
                                       size_t header_length, void *data,
                                       size_t length,
                                       const ucp_am_recv_param_t *param)
    {
        test_ucp_am_nbx *self = reinterpret_cast<test_ucp_am_nbx*>(arg);
        self->m_am_received   = true;
        return UCS_OK;
    }

    static ucs_status_t am_data_hold_cb(void *arg, const void *header,
                                        size_t header_length, void *data,
                                        size_t length,
                                        const ucp_am_recv_param_t *param)
    {
        void **rx_data_p = reinterpret_cast<void**>(arg);

        EXPECT_TRUE(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA);
        EXPECT_EQ(NULL, *rx_data_p);

        *rx_data_p = data;

        return UCS_INPROGRESS;
    }

    static void am_data_recv_cb(void *request, ucs_status_t status,
                                size_t length, void *user_data)
    {
        test_ucp_am_nbx *self = reinterpret_cast<test_ucp_am_nbx*>(user_data);

        EXPECT_UCS_OK(status);

        self->am_recv_check_data(length);
    }

    static const uint16_t           TEST_AM_NBX_ID = 0;
    ucp_datatype_t                  m_dt;
    volatile bool                   m_am_received;
    std::string                     m_hdr;
    ucp_datatype_t                  m_rx_dt;
    ucs_memory_type_t               m_rx_memtype;
    ucp::data_type_desc_t           m_rx_dt_desc;
    void                            *m_rx_buf;
};

UCS_TEST_P(test_ucp_am_nbx, set_invalid_handler)
{
    ucp_am_handler_param_t params;

    params.id           = 0;
    params.cb           = NULL;
    params.field_mask   = UCP_AM_HANDLER_PARAM_FIELD_ID |
                          UCP_AM_HANDLER_PARAM_FIELD_CB;
    ucs_status_t status = ucp_worker_set_am_recv_handler(sender().worker(),
                                                         &params);
    EXPECT_UCS_OK(status);

    // Check that error is returned if id or callback is not set
    scoped_log_handler wrap_err(wrap_errors_logger);

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_CB;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    params.field_mask = 0ul;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    // Check that error is returned if private flag is requested by the user
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                        UCP_AM_HANDLER_PARAM_FIELD_CB |
                        UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
    params.flags      = UCP_AM_CB_PRIV_FLAG_NBX;
    status            = ucp_worker_set_am_recv_handler(sender().worker(),
                                                       &params);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_ucp_am_nbx, max_am_header)
{
    size_t min_am_bcopy       = std::numeric_limits<size_t>::max();
    bool has_tl_with_am_bcopy = false;

    for (ucp_rsc_index_t idx = 0; idx < sender().ucph()->num_tls; ++idx) {
        uct_iface_attr_t *attr = ucp_worker_iface_get_attr(sender().worker(), idx);
        if (attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
            min_am_bcopy = ucs_min(min_am_bcopy, attr->cap.am.max_bcopy);
            has_tl_with_am_bcopy = true;
        }
    }

    EXPECT_TRUE(has_tl_with_am_bcopy);

    EXPECT_GE(max_am_hdr(), 64ul);
    EXPECT_LT(max_am_hdr(), min_am_bcopy);
}

UCS_TEST_P(test_ucp_am_nbx, zero_send)
{
    test_am_send_recv(0, max_am_hdr());
}

UCS_TEST_P(test_ucp_am_nbx, rx_persistent_data)
{
    void *rx_data = NULL;
    char data     = 'd';

    set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_hold_cb, &rx_data,
                        UCP_AM_FLAG_PERSISTENT_DATA);

    ucp_request_param_t param;

    param.op_attr_mask    = 0ul;
    ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID, NULL,
                                            0ul, &data, sizeof(data), &param);
    wait_for_flag(&rx_data);
    EXPECT_TRUE(rx_data != NULL);
    EXPECT_EQ(data, *reinterpret_cast<char*>(rx_data));

    ucp_am_data_release(receiver().worker(), rx_data);
    EXPECT_EQ(UCS_OK, request_wait(sptr));
}

// Check that max_short limits are adjusted when rndv threshold is set
UCS_TEST_P(test_ucp_am_nbx, max_short_thresh_rndv, "RNDV_THRESH=0")
{
    ucp_ep_config_t *ep_cfg = ucp_ep_config(sender().ep());

    size_t max_short = static_cast<size_t>(
            ep_cfg->am_u.max_eager_short.memtype_on + 1);

    test_short_thresh(max_short);

    size_t max_reply_short = static_cast<size_t>(
            ep_cfg->am_u.max_reply_eager_short.memtype_on + 1);

    test_short_thresh(max_reply_short);
}

// Check that max_short limits are adjusted when zcopy threshold is set
UCS_TEST_P(test_ucp_am_nbx, max_short_thresh_zcopy, "ZCOPY_THRESH=0")
{
    ucp_ep_config_t *ep_cfg = ucp_ep_config(sender().ep());

    size_t max_short = static_cast<size_t>(
            ep_cfg->am_u.max_eager_short.memtype_on + 1);

    EXPECT_LE(max_short, ep_cfg->am.zcopy_thresh[0]);


    size_t max_reply_short = static_cast<size_t>(
            ep_cfg->am_u.max_reply_eager_short.memtype_on + 1);

    EXPECT_LE(max_reply_short, ep_cfg->am.zcopy_thresh[0]);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx)


class test_ucp_am_nbx_closed_ep : public test_ucp_am_nbx {
protected:
    virtual ucp_ep_params_t get_ep_params()
    {
        ucp_ep_params_t ep_params = test_ucp_am_nbx::get_ep_params();
        ep_params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        /* The error handling requirement is needed since we need to take care of
         * a case when a receiver tries to fetch data on a closed EP */
        ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        return ep_params;
    }

    void test_recv_on_closed_ep(size_t size, unsigned flags = 0,
                                bool poke_rx_progress = false,
                                bool rx_expected = false)
    {
        skip_loopback();
        test_am_send_recv(0, max_am_hdr()); // warmup wireup

        m_am_received = false;
        std::vector<char> sbuf(size, 'd');
        ucp::data_type_desc_t sdt_desc(m_dt, &sbuf[0], size);

        set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_rx_check_cb, this);

        ucs_status_ptr_t sreq = send_am(sdt_desc, flags);

        sender().progress();
        if (poke_rx_progress) {
            receiver().progress();
            if (m_am_received) {
                request_wait(sreq);
                UCS_TEST_SKIP_R("received all AMs before ep closed");
            }
        }

        void *close_req = receiver().disconnect_nb(0, 0,
                                                   UCP_EP_CLOSE_MODE_FLUSH);
        ucs_time_t deadline = ucs::get_deadline(10);
        while (!is_request_completed(close_req) &&
               (ucs_get_time() < deadline)) {
            progress();
        };

        receiver().close_ep_req_free(close_req);

        if (rx_expected) {
            request_wait(sreq);
            wait_for_flag(&m_am_received);
        } else {
            // Send request may complete with error
            // (rndv should complete with EP_TIMEOUT)
            scoped_log_handler wrap_err(wrap_errors_logger);
            request_wait(sreq);
        }

        EXPECT_EQ(rx_expected, m_am_received);
    }
};


UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_short_am_on_closed_ep, "RNDV_THRESH=inf")
{
    // Single fragment message sent without REPLY flag is expected
    // to be received even if remote side closes its ep
    test_recv_on_closed_ep(8, 0, false, true);
}

// All the following type of AM messages are expected to be dropped on the
// receiver side, when its ep is closed
UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_short_reply_am_on_closed_ep, "RNDV_THRESH=inf")
{
    test_recv_on_closed_ep(8, UCP_AM_SEND_REPLY);
}

UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_long_am_on_closed_ep, "RNDV_THRESH=inf")
{
    test_recv_on_closed_ep(64 * UCS_KBYTE, 0, true);
}

UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_long_reply_am_on_closed_ep, "RNDV_THRESH=inf")
{
    test_recv_on_closed_ep(64 * UCS_KBYTE, UCP_AM_SEND_REPLY, true);
}

UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_rts_am_on_closed_ep, "RNDV_THRESH=32K")
{
    test_recv_on_closed_ep(64 * UCS_KBYTE, 0);
}

UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_rts_reply_am_on_closed_ep, "RNDV_THRESH=32K")
{
    test_recv_on_closed_ep(64 * UCS_KBYTE, UCP_AM_SEND_REPLY);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_closed_ep)


class test_ucp_am_nbx_eager_memtype : public test_ucp_am_nbx {
public:
    void init()
    {
        modify_config("RNDV_THRESH", "inf");
        test_ucp_am_nbx::init();
        m_rx_memtype = static_cast<ucs_memory_type_t>(get_variant_value(1));
    }

    static void base_test_generator(std::vector<ucp_test_variant> &variants)
    {
        // 1. Do not instantiate test case if no GPU memtypes supported.
        // 2. Do not exclude host memory type, because this generator is used by
        //    test_ucp_am_nbx_rndv_memtype class to generate combinations like
        //    host<->cuda, cuda-managed<->host, etc.
        if (!mem_buffer::is_gpu_supported()) {
            return;
        }

        add_variant_memtypes(variants, test_ucp_am_base::get_test_variants,
                             std::numeric_limits<uint64_t>::max());
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_memtypes(variants, base_test_generator,
                             std::numeric_limits<uint64_t>::max());
    }
};

UCS_TEST_P(test_ucp_am_nbx_eager_memtype, basic)
{
    ucs_memory_type_t mt = static_cast<ucs_memory_type_t>(get_variant_value(0));
    test_am_send_recv(16 * UCS_KBYTE, 8, 0, mt);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_am_nbx_eager_memtype)


class test_ucp_am_nbx_eager_data_release : public test_ucp_am_nbx {
public:
    test_ucp_am_nbx_eager_data_release()
    {
        modify_config("RNDV_THRESH", "inf");
        modify_config("ZCOPY_THRESH", "inf");
        m_data_ptr = NULL;
    }

    virtual ucs_status_t
    am_data_handler(const void *header, size_t header_length, void *data,
                    size_t length, const ucp_am_recv_param_t *rx_param)
    {
        EXPECT_FALSE(m_am_received);
        EXPECT_TRUE(rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA);

        m_am_received = true;
        m_data_ptr    = data;

        check_header(header, header_length);
        mem_buffer::pattern_check(data, length, SEED);

        return UCS_INPROGRESS;
    }

    void test_data_release(size_t size)
    {
        size_t hdr_size = ucs_min(max_am_hdr(), 8);
        test_am_send_recv(size, 0, 0, UCS_MEMORY_TYPE_HOST,
                          UCP_AM_FLAG_PERSISTENT_DATA);
        ucp_am_data_release(receiver().worker(), m_data_ptr);

        test_am_send_recv(size, hdr_size, 0, UCS_MEMORY_TYPE_HOST,
                          UCP_AM_FLAG_PERSISTENT_DATA);
        ucp_am_data_release(receiver().worker(), m_data_ptr);
    }

    size_t fragment_size()
    {
        return ucp_ep_config(sender().ep())->am.max_bcopy -
               sizeof(ucp_am_hdr_t);
    }

private:
    void *m_data_ptr;
};

UCS_TEST_P(test_ucp_am_nbx_eager_data_release, short)
{
    test_data_release(1);
}

UCS_TEST_P(test_ucp_am_nbx_eager_data_release, single)
{
    test_data_release(fragment_size() / 2);
}

UCS_TEST_P(test_ucp_am_nbx_eager_data_release, multi)
{
    test_data_release(fragment_size() * 2);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_eager_data_release)


class test_ucp_am_nbx_dts : public test_ucp_am_nbx {
public:
    static const uint64_t dts_bitmap = UCS_BIT(UCP_DATATYPE_CONTIG) |
                                       UCS_BIT(UCP_DATATYPE_IOV) |
                                       UCS_BIT(UCP_DATATYPE_GENERIC);

    virtual ucp_ep_params_t get_ep_params()
    {
        ucp_ep_params_t ep_params = test_ucp_am_nbx::get_ep_params();

        ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.err_mode    = static_cast<ucp_err_handling_mode_t>(
                                                          get_variant_value(3));
        return ep_params;
    }

    static void get_test_dts(std::vector<ucp_test_variant>& variants)
    {
        /* coverity[overrun-buffer-val] */
        add_variant_values(variants, test_ucp_am_base::get_test_variants,
                           dts_bitmap, ucp_datatype_class_names);
    }

    static void base_test_generator(std::vector<ucp_test_variant> &variants)
    {
        /* push variant for the receive type, on top of existing dts variants */
        /* coverity[overrun-buffer-val] */
        add_variant_values(variants, get_test_dts, dts_bitmap,
                           ucp_datatype_class_names);
    }

    static void get_test_dts_reply(std::vector<ucp_test_variant>& variants)
    {
        add_variant_values(variants, base_test_generator, 0);
        add_variant_values(variants, base_test_generator, UCP_AM_SEND_REPLY,
                           "reply");
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant_values(variants, get_test_dts_reply,
                           UCP_ERR_HANDLING_MODE_NONE);
        add_variant_values(variants, get_test_dts_reply,
                           UCP_ERR_HANDLING_MODE_PEER, "errh");
    }

    void init()
    {
        test_ucp_am_nbx::init();

        m_dt    = make_dt(get_variant_value(0));
        m_rx_dt = make_dt(get_variant_value(1));
    }

    void cleanup()
    {
        destroy_dt(m_dt);
        destroy_dt(m_rx_dt);
        test_ucp_am_nbx::cleanup();
    }

    virtual unsigned get_send_flag()
    {
        return get_variant_value(2);
    }

    virtual ucs_status_t
    am_data_handler(const void *header, size_t header_length, void *data,
                    size_t length, const ucp_am_recv_param_t *rx_param)
    {
        EXPECT_FALSE(rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV);

        return test_ucp_am_nbx::am_data_handler(header, header_length, data,
                                                length, rx_param);
    }
};

UCS_TEST_P(test_ucp_am_nbx_dts, short_send)
{
    test_am(1);
}

UCS_TEST_P(test_ucp_am_nbx_dts, short_bcopy_send, "ZCOPY_THRESH=-1",
                                                  "RNDV_THRESH=-1")
{
    test_am(4 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_dts, long_bcopy_send, "ZCOPY_THRESH=-1",
                                                 "RNDV_THRESH=-1")
{
    test_am(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_dts, short_zcopy_send, "ZCOPY_THRESH=1",
                                                  "RNDV_THRESH=-1")
{
    test_am(4 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_dts, long_zcopy_send, "ZCOPY_THRESH=1",
                                                 "RNDV_THRESH=-1")
{
    test_am(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_dts, send_eager_flag, "RNDV_THRESH=128")
{
    test_am(64 * UCS_KBYTE, UCP_AM_SEND_FLAG_EAGER);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_dts)


class test_ucp_am_nbx_rndv : public test_ucp_am_nbx {
public:
    struct am_cb_args {
        test_ucp_am_nbx_rndv *self;
        void                 **desc;
    };

    test_ucp_am_nbx_rndv()
    {
        m_status = UCS_OK;
        modify_config("RNDV_THRESH", "128");
    }

    ucs_status_t am_data_handler(const void *header, size_t header_length,
                                 void *data, size_t length,
                                 const ucp_am_recv_param_t *rx_param)
    {
        EXPECT_TRUE(rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV);
        EXPECT_FALSE(rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA);

        ucs_status_t status = test_ucp_am_nbx::am_data_handler(header,
                                                               header_length,
                                                               data, length,
                                                               rx_param);
        EXPECT_FALSE(UCS_STATUS_IS_ERR(status));

        return UCS_INPROGRESS;
    }

    static ucs_status_t am_data_reject_rndv_cb(void *arg, const void *header,
                                               size_t header_length, void *data,
                                               size_t length,
                                               const ucp_am_recv_param_t *param)
    {
        test_ucp_am_nbx_rndv *self = reinterpret_cast<test_ucp_am_nbx_rndv*>(arg);

        EXPECT_FALSE(self->m_am_received);
        self->m_am_received = true;

        return self->m_status;
    }

    static ucs_status_t am_data_deferred_reject_rndv_cb(void *arg,
                                                        const void *header,
                                                        size_t header_length,
                                                        void *data, size_t length,
                                                        const ucp_am_recv_param_t *param)
    {
        void **data_desc_p = reinterpret_cast<void**>(arg);

        EXPECT_EQ(NULL, *data_desc_p);
        *data_desc_p = data;

        return UCS_INPROGRESS;
    }

    static ucs_status_t am_data_drop_rndv_cb(void *arg,
                                             const void *header,
                                             size_t header_length,
                                             void *data, size_t length,
                                             const ucp_am_recv_param_t *param)
    {
        struct am_cb_args *args    = reinterpret_cast<am_cb_args*>(arg);
        test_ucp_am_nbx_rndv *self = args->self;
        void **data_desc_p         = args->desc;

        *data_desc_p = data;
        self->m_am_received = true;

        /* return UCS_OK without calling ucp_am_recv_data_nbx()
         * to drop the message */
        return UCS_OK;
    }

    ucs_status_t m_status;
};

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_auto, "RNDV_SCHEME=auto")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_get, "RNDV_SCHEME=get_zcopy")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_put, "RNDV_SCHEME=put_zcopy")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_am, "RNDV_SCHEME=am")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_flag_zero_send, "RNDV_THRESH=inf")
{
    test_am_send_recv(0, 0, UCP_AM_SEND_FLAG_RNDV);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_flag_send, "RNDV_THRESH=inf")
{
    test_am_send_recv(64 * UCS_KBYTE, 0, UCP_AM_SEND_FLAG_RNDV);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_zero_send, "RNDV_THRESH=0")
{
    test_am_send_recv(0);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, just_header_rndv, "RNDV_THRESH=1")
{
    test_am_send_recv(0, max_am_hdr());
}

UCS_TEST_P(test_ucp_am_nbx_rndv, header_and_data_rndv, "RNDV_THRESH=128")
{
    test_am_send_recv(127, 1);
}

UCS_TEST_SKIP_COND_P(test_ucp_am_nbx_rndv, invalid_recv_desc,
                     RUNNING_ON_VALGRIND, "RNDV_THRESH=1")
{
    void *data_desc = NULL;
    void *rx_data   = NULL;
    char data       = 'd';
    ucp_request_param_t param;

    struct am_cb_args args = { this,  &data_desc };
    set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_drop_rndv_cb, &args);

    param.op_attr_mask = 0ul;

    ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID, NULL,
                                            0ul, &data, sizeof(data), &param);

    wait_for_flag(&m_am_received);

    scoped_log_handler wrap_err(wrap_errors_logger);
    /* attempt to recv data with invalid 'data_desc' since it was reliased
     * due to am_data_drop_rndv_cb() returned UCS_OK */
    ucs_status_ptr_t rptr = ucp_am_recv_data_nbx(receiver().worker(),
                                                 data_desc,
                                                 rx_data, sizeof(data),
                                                 &param);

    EXPECT_EQ(UCS_ERR_INVALID_PARAM, UCS_PTR_STATUS(rptr));

    request_wait(sptr);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, reject_rndv)
{
    skip_loopback();

    set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_reject_rndv_cb,
                        this);

    std::vector<char> sbuf(10000, 0);
    ucp_request_param_t param;
    param.op_attr_mask      = 0ul;
    ucs_status_t statuses[] = {UCS_OK, UCS_ERR_REJECTED, UCS_ERR_NO_MEMORY};

    scoped_log_handler wrap_err(wrap_errors_logger);

    for (int i = 0; i < ucs_static_array_size(statuses); ++i) {
        m_am_received = false;
        m_status      = statuses[i];

        ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID,
                                                NULL, 0ul, sbuf.data(),
                                                sbuf.size(), &param);

        EXPECT_EQ(m_status, request_wait(sptr));
        EXPECT_TRUE(m_am_received);
    }
}

UCS_TEST_P(test_ucp_am_nbx_rndv, deferred_reject_rndv)
{
    skip_loopback();

    void *data_desc = NULL;
    std::vector<char> sbuf(10000, 0);
    ucp_request_param_t param;

    param.op_attr_mask = 0ul;

    set_am_data_handler(receiver(), TEST_AM_NBX_ID,
                        am_data_deferred_reject_rndv_cb, &data_desc);

    ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID,
                                            NULL, 0ul, sbuf.data(),
                                            sbuf.size(), &param);

    wait_for_flag(&data_desc);
    EXPECT_TRUE(data_desc != NULL);

    ucp_am_data_release(receiver().worker(), data_desc);
    EXPECT_EQ(UCS_OK, request_wait(sptr));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_rndv)


class test_ucp_am_nbx_rndv_dts : public test_ucp_am_nbx_rndv {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        /* push variant for the receive type, on top of existing dts variants */
        /* coverity[overrun-buffer-val] */
        add_variant_values(variants, test_ucp_am_nbx_dts::get_test_dts,
                           test_ucp_am_nbx_dts::dts_bitmap,
                           ucp_datatype_class_names);
    }

    void init()
    {
        test_ucp_am_nbx::init();

        m_dt    = make_dt(get_variant_value(0));
        m_rx_dt = make_dt(get_variant_value(1));
    }

    void cleanup()
    {
        destroy_dt(m_dt);
        destroy_dt(m_rx_dt);

        test_ucp_am_nbx::cleanup();
    }
};

UCS_TEST_P(test_ucp_am_nbx_rndv_dts, rndv, "RNDV_THRESH=256")
{
    test_am_send_recv(64 * UCS_KBYTE);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_rndv_dts);


class test_ucp_am_nbx_rndv_memtype : public test_ucp_am_nbx_rndv {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        // Test will not be instantiated if no GPU memtypes supported, because
        // of the check for supported memory types in
        // test_ucp_am_nbx_eager_memtype::get_test_variants
        return test_ucp_am_nbx_eager_memtype::get_test_variants(variants);
    }

    void init()
    {
        modify_config("RNDV_THRESH", "128");

        test_ucp_am_nbx::init();
        m_rx_memtype = static_cast<ucs_memory_type_t>(get_variant_value(1));
    }
};

UCS_TEST_P(test_ucp_am_nbx_rndv_memtype, rndv)
{
    ucs_memory_type_t mt = static_cast<ucs_memory_type_t>(get_variant_value(0));
    test_am_send_recv(64 * UCS_KBYTE, 8, 0, mt);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_am_nbx_rndv_memtype);
