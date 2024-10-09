/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2020. ALL RIGHTS RESERVED.
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
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_am.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_resource.h>
#include <ucs/datastruct/mpool.inl>
}

#define NUM_MESSAGES 17

#define UCP_REALLOC_ID 1000
#define UCP_SEND_ID    0
#define UCP_REPLY_ID   1
#define UCP_RELEASE    1

class test_ucp_am_base : public ucp_test {
public:
    test_ucp_am_base()
    {
        if (get_variant_value()) {
            modify_config("PROTO_ENABLE", "n");
        }
    }

    static void get_test_variants(variant_vec_t &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_AM, 0, "");
        if (!RUNNING_ON_VALGRIND) {
            add_variant_with_value(variants, UCP_FEATURE_AM, 1, "proto_v1");
        }
    }

    virtual void init()
    {
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
        for (int i = 0; i < recv_ams; i++) {
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
    do_send_process_data_test(0, UCP_SEND_ID, UCP_AM_SEND_FLAG_REPLY);
}

UCS_TEST_P(test_ucp_am, send_process_am_rndv, "RNDV_THRESH=1")
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(0, UCP_SEND_ID, 0);

    set_reply_handlers();
    do_send_process_data_test(0, UCP_SEND_ID, UCP_AM_SEND_FLAG_REPLY);
}

UCS_TEST_P(test_ucp_am, send_process_am_release)
{
    set_handlers(UCP_SEND_ID);
    do_send_process_data_test(UCP_RELEASE, 0, 0);
}

UCS_TEST_P(test_ucp_am, send_process_iov_am_64k_size)
{
    do_send_process_data_iov_test(65536);
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

UCS_TEST_P(test_ucp_am, set_am_handler_out_of_order)
{
    set_handlers(UCP_SEND_ID + 20);
    set_handlers(UCP_SEND_ID);
    set_handlers(UCP_SEND_ID + 10);

    do_send_process_data_test(0, UCP_SEND_ID, 0);
    do_send_process_data_test(0, UCP_SEND_ID + 10, 0);
    do_send_process_data_test(0, UCP_SEND_ID + 20, 0);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am)


class test_ucp_am_nbx : public test_ucp_am_base {
public:
    static const uint64_t SEED = 0x1111111111111111lu;

    test_ucp_am_nbx()
    {
        reset_counters();
        m_dt      = ucp_dt_make_contig(1);
        m_rx_dt   = ucp_dt_make_contig(1);
        m_rx_buf  = NULL;
        m_rx_memh = NULL;
    }

    void test_datatypes(std::function<void()> test_f,
                        const std::vector<ucp_dt_type> &datatypes =
                        {UCP_DATATYPE_CONTIG,
                         UCP_DATATYPE_IOV,
                         UCP_DATATYPE_GENERIC});

    void skip_no_am_lane_caps(uint64_t caps, const std::string &str);

protected:
    virtual ucs_memory_type_t tx_memtype() const
    {
        return UCS_MEMORY_TYPE_HOST;
    }

    virtual ucs_memory_type_t rx_memtype() const
    {
        return UCS_MEMORY_TYPE_HOST;
    }

    void reset_counters()
    {
        m_send_counter = 0;
        m_recv_counter = 0;
    }

    void wait_receives()
    {
        wait_for_value(&m_recv_counter, m_send_counter);
    }

    size_t max_am_hdr()
    {
        ucp_worker_attr_t attr;
        attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;

        ASSERT_UCS_OK(ucp_worker_query(sender().worker(), &attr));
        return attr.max_am_header;
    }

    size_t fragment_size()
    {
        return ucp_ep_config(sender().ep())->am.max_bcopy -
               sizeof(ucp_am_hdr_t);
    }

    virtual unsigned get_send_flag() const
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
            ucs_assertv(UCP_DATATYPE_GENERIC == dt, "dt=%d", dt);
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

    ucs_status_t set_am_data_handler_internal(entity &e, unsigned am_id,
                                              ucp_am_recv_callback_t cb, void *arg,
                                              unsigned flags = 0)
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

        return ucp_worker_set_am_recv_handler(e.worker(), &param);
    }

    void set_am_data_handler(entity &e, uint16_t am_id,
                             ucp_am_recv_callback_t cb, void *arg,
                             unsigned flags = 0)
    {
        ASSERT_UCS_OK(set_am_data_handler_internal(e, am_id, cb, arg, flags));
    }

    void check_header(const void *header, size_t header_length)
    {
        std::string check_pattern((char*)header, header_length);
        EXPECT_EQ(check_pattern, m_hdr);
    }

    ucs_status_ptr_t
    update_counter_and_send_am(const void *header, size_t header_length,
                               const void *buffer, size_t count, unsigned am_id,
                               const ucp_request_param_t *param)
    {
        m_send_counter++;
        return ucp_am_send_nbx(sender().ep(), am_id, header,
                               header_length, buffer, count, param);
    }

    ucs_status_ptr_t
    send_am(const ucp::data_type_desc_t &dt_desc, unsigned flags = 0,
            const void *hdr = NULL, unsigned hdr_length = 0,
            const ucp_mem_h memh = NULL, uint32_t op_attr_mask = 0)
    {
        ucp_request_param_t param;
        param.op_attr_mask = op_attr_mask | UCP_OP_ATTR_FIELD_DATATYPE;
        param.datatype     = dt_desc.dt();

        if (flags != 0) {
            param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
            param.flags         = flags;
        }

        if (memh != NULL) {
            param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            param.memh          = memh;
        }

        ucs_status_ptr_t sptr = update_counter_and_send_am(hdr, hdr_length,
                                                           dt_desc.buf(),
                                                           dt_desc.count(),
                                                           TEST_AM_NBX_ID,
                                                           &param);
        return sptr;
    }

    void test_am_send_recv(size_t size, size_t header_size = 0ul,
                           unsigned flags = 0, unsigned data_cb_flags = 0,
                           uint32_t op_attr_mask = 0)
    {
        mem_buffer sbuf(size, tx_memtype());
        sbuf.pattern_fill(SEED);
        m_hdr.resize(header_size);
        ucs::fill_random(m_hdr);
        reset_counters();
        ucp_mem_h memh = NULL;

        set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_cb, this,
                            data_cb_flags);

        ucp::data_type_desc_t sdt_desc(m_dt, sbuf.ptr(), size);

        if (prereg()) {
            memh = sender().mem_map(sbuf.ptr(), size);
        }

        ucs_status_ptr_t sptr = send_am(sdt_desc, get_send_flag() | flags,
                                        m_hdr.data(), m_hdr.size(), memh,
                                        op_attr_mask);

        wait_receives();
        request_wait(sptr);

        if (prereg()) {
            sender().mem_unmap(memh);
        }

        EXPECT_EQ(m_recv_counter, m_send_counter);
    }

    void test_am_send_recv_memtype(size_t size, size_t header_size = 8)
    {
        std::vector<ucp_dt_type> dts = {UCP_DATATYPE_CONTIG};

        if (is_proto_enabled()) {
            dts.push_back(UCP_DATATYPE_IOV);
        }

        test_datatypes([&]() { test_am_send_recv(size, header_size); }, dts);
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

    ucs_status_t am_data_rndv_handler(void *data, size_t length)
    {
        ucs_status_t status;

        m_rx_buf = mem_buffer::allocate(length, rx_memtype());
        mem_buffer::pattern_fill(m_rx_buf, length, 0ul, rx_memtype());

        m_rx_dt_desc.make(m_rx_dt, m_rx_buf, length);

        uint32_t imm_compl_flag = UCP_OP_ATTR_FLAG_NO_IMM_CMPL *
                                  (ucs::rand() % 2);
        size_t rx_length        = SIZE_MAX;
        ucp_request_param_t params;
        params.op_attr_mask     = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_USER_DATA |
                                  UCP_OP_ATTR_FIELD_DATATYPE |
                                  UCP_OP_ATTR_FIELD_RECV_INFO |
                                  imm_compl_flag;
        params.datatype         = m_rx_dt_desc.dt();
        params.cb.recv_am       = am_data_recv_cb;
        params.user_data        = this;
        params.recv_info.length = &rx_length;

        if (prereg()) {
            params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
            m_rx_memh            = receiver().mem_map(m_rx_buf, length);
            params.memh          = m_rx_memh;
        }

        ucs_status_ptr_t sp = ucp_am_recv_data_nbx(receiver().worker(),
                                                   data, m_rx_dt_desc.buf(),
                                                   m_rx_dt_desc.count(),
                                                   &params);
        if (UCS_PTR_IS_PTR(sp)) {
            ucp_request_release(sp);
            status = UCS_INPROGRESS;
        } else {
            EXPECT_EQ(NULL, sp);
            EXPECT_EQ(length, rx_length);
            am_recv_check_data(rx_length);
            status = UCS_OK;
        }

        return status;
    }

    virtual ucs_status_t am_data_handler(const void *header,
                                         size_t header_length,
                                         void *data, size_t length,
                                         const ucp_am_recv_param_t *rx_param)
    {
        EXPECT_LT(m_recv_counter, m_send_counter);
        check_header(header, header_length);

        bool has_reply_ep = get_send_flag() & UCP_AM_SEND_FLAG_REPLY;

        EXPECT_EQ(has_reply_ep,
                  !!(rx_param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP));
        EXPECT_EQ(has_reply_ep, rx_param->reply_ep != NULL);

        if (!(rx_param->recv_attr &
              (UCP_AM_RECV_ATTR_FLAG_RNDV | UCP_AM_RECV_ATTR_FLAG_DATA))) {
            mem_buffer::pattern_check(data, length, SEED);
            m_recv_counter++;
            return UCS_OK;
        }

        return am_data_rndv_handler(data, length);
    }


    void am_recv_check_data(size_t length)
    {
        EXPECT_LT(m_recv_counter, m_send_counter);
        mem_buffer::pattern_check(m_rx_buf, length, SEED, rx_memtype());
        m_recv_counter++;

        if (m_rx_memh != NULL) {
            receiver().mem_unmap(m_rx_memh);
        }
        mem_buffer::release(m_rx_buf, rx_memtype());
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
        self->m_recv_counter++;
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

    virtual bool prereg() const
    {
        return 0;
    }

    static const uint16_t           TEST_AM_NBX_ID = 0;
    volatile size_t                 m_send_counter;
    volatile size_t                 m_recv_counter;
    ucp_datatype_t                  m_dt;
    std::string                     m_hdr;
    ucp_datatype_t                  m_rx_dt;
    ucp::data_type_desc_t           m_rx_dt_desc;
    void                            *m_rx_buf;
    ucp_mem_h                       m_rx_memh;
};

class test_ucp_am_id : public test_ucp_am_nbx {
protected:
    void reset_counters()
    {
        test_ucp_am_nbx::reset_counters();
        m_recv_counter_cb = 0;
    }

    void test_am_id_handler()
    {
        reset_counters();

        ucs_status_t status = set_am_data_handler_internal(
                                            receiver(), 0xffff0001,
                                            am_id_overflow_data_cb, this);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

        ucp_request_param_t param;
        param.op_attr_mask = 0;

        ucs_status_ptr_t sptr = update_counter_and_send_am(NULL, 0ul, NULL, 0,
                                                           0xffff0001, &param);
        EXPECT_EQ(UCS_PTR_STATUS(sptr), UCS_ERR_INVALID_PARAM);
        EXPECT_EQ(0, m_recv_counter_cb);
    }

    static ucs_status_t am_id_overflow_data_cb(void *arg, const void *header,
                                               size_t header_length,
                                               void *data, size_t length,
                                               const ucp_am_recv_param_t *param)
    {
        test_ucp_am_id *self = reinterpret_cast<test_ucp_am_id*>(arg);
        self->m_recv_counter_cb++;
        return self->am_data_handler(header, header_length,
                                     data, length, param);
    }

    volatile size_t m_recv_counter_cb;
};

void test_ucp_am_nbx::test_datatypes(std::function<void()> test_f,
                                     const std::vector<ucp_dt_type> &datatypes)
{
    for (const auto &dt_it : datatypes) {
        m_dt = make_dt(dt_it);

        for (const auto &rx_dt_it : datatypes) {
            m_rx_dt = make_dt(rx_dt_it);
            test_f();
            destroy_dt(m_rx_dt);
        }

        destroy_dt(m_dt);
    }
}

void test_ucp_am_nbx::skip_no_am_lane_caps(uint64_t caps,
                                           const std::string &reason)
{
    ucp_ep_config_key_t key  = ucp_ep_config(sender().ep())->key;
    ucp_lane_index_t am_lane = key.am_lane;
    if (am_lane == UCP_NULL_LANE) {
        UCS_TEST_SKIP_R("am lane is null");
    }

    uint64_t iface_caps = ucp_worker_iface_get_attr(
            sender().worker(), key.lanes[am_lane].rsc_index)->cap.flags;
    if (!ucs_test_all_flags(iface_caps, caps)) {
        UCS_TEST_SKIP_R(reason);
    }
}


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

#if ENABLE_PARAMS_CHECK
UCS_TEST_P(test_ucp_am_nbx, am_header_error)
{
    scoped_log_handler wrap_err(wrap_errors_logger);

    ucp_request_param_t param;
    param.op_attr_mask    = 0ul;
    ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID, NULL,
                                            max_am_hdr() + 1, NULL, 0, &param);
    EXPECT_EQ(UCS_PTR_STATUS(sptr), UCS_ERR_INVALID_PARAM);
}
#endif

UCS_TEST_P(test_ucp_am_id, am_id_overflow)
{
    scoped_log_handler wrap_err(wrap_errors_logger);
    test_am_id_handler();
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_id)

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

UCS_TEST_P(test_ucp_am_nbx, rx_am_mpools,
           "RX_MPOOL_SIZES=2,8,64,128", "RNDV_THRESH=inf")
{
    void *rx_data = NULL;

    set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_hold_cb, &rx_data,
                        UCP_AM_FLAG_PERSISTENT_DATA);

    static const std::string ib_tls[] = { "dc_x", "rc_v", "rc_x", "ud_v",
                                          "ud_x", "ib" };

    // UCP takes desc from mpool only for data arrived as inlined from UCT.
    // Typically, with IB, data is inlined up to 32 bytes, so use smaller range
    // of values for IB transports.
    bool has_ib = has_any_transport(
            std::vector<std::string>(ib_tls,
                                     ib_tls + ucs_static_array_size(ib_tls)));
    ssize_t length = ucs::rand() % (has_ib ? 32 : 256);
    std::vector<char> sbuf(length, 'd');

    ucp_request_param_t param;
    param.op_attr_mask = 0ul;

    ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID, NULL,
                                            0ul, sbuf.data(), sbuf.size(),
                                            &param);
    wait_for_flag(&rx_data);
    EXPECT_TRUE(rx_data != NULL);
    EXPECT_EQ(UCS_OK, request_wait(sptr));

    ucp_recv_desc_t *rdesc = (ucp_recv_desc_t*)rx_data - 1;
    if (rdesc->flags & UCP_RECV_DESC_FLAG_UCT_DESC) {
        ucp_am_data_release(receiver().worker(), rx_data);
        UCS_TEST_SKIP_R("non-inline data arrived");
    } else {
        UCS_TEST_MESSAGE << "length " << length;
    }

    ucp_worker_h worker = receiver().worker();

    for (int i = 0; i < ucs_popcount(worker->am_mps.bitmap); ++i) {
        ucs_mpool_t *mpool =
            &reinterpret_cast<ucs_mpool_t*>(worker->am_mps.data)[i];
        ssize_t elem_size  = mpool->data->elem_size - (sizeof(ucs_mpool_elem_t) +
                             UCP_WORKER_HEADROOM_SIZE + worker->am.alignment);
        ASSERT_TRUE(elem_size >= 0);

        if (elem_size >= (length + 1)) {
            EXPECT_EQ(ucs_mpool_obj_owner(rdesc), mpool);
            break;
        }

        EXPECT_NE(ucs_mpool_obj_owner(rdesc), mpool);
    }

    ucp_am_data_release(receiver().worker(), rx_data);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx)


class test_ucp_am_nbx_reply_always : public test_ucp_am_nbx {
protected:
    virtual unsigned get_send_flag() const
    {
        return UCP_AM_SEND_FLAG_REPLY;
    }
};

/* The following two tests, "multi_bcopy" and "multi_zcopy", check correctness
 * of AM API when using UCP_AM_SEND_FLAG_REPLY flag.
 * Tests send messages with size equal to (fragment size + 1). The size should
 * be less than (fragment size + reply footer size) to check if the switch
 * between single and multi protocols is correct. */
UCS_TEST_P(test_ucp_am_nbx_reply_always, multi_bcopy, "ZCOPY_THRESH=inf",
           "RNDV_THRESH=inf")
{
    skip_no_am_lane_caps(UCT_IFACE_FLAG_AM_BCOPY, "am_bcopy is not supported");
    size_t bcopy_fragment_size = fragment_size() - sizeof(ucp_am_reply_ftr_t);
    test_am_send_recv(bcopy_fragment_size + 1, 0);
}

UCS_TEST_P(test_ucp_am_nbx_reply_always, multi_zcopy, "ZCOPY_THRESH=1",
           "RNDV_THRESH=inf")
{
    skip_no_am_lane_caps(UCT_IFACE_FLAG_AM_ZCOPY, "am_zcopy is not supported");
    size_t zcopy_fragment_size = ucp_ep_config(sender().ep())->am.max_zcopy -
                                 sizeof(ucp_am_hdr_t) -
                                 sizeof(ucp_am_reply_ftr_t);
    test_am_send_recv(zcopy_fragment_size + 1, 0);
}

UCS_TEST_P(test_ucp_am_nbx_reply_always, short_slow_path)
{
    /*
     * This message is sent with UCP_OP_ATTR_FLAG_NO_IMM_CMPL so it will
     * always go through AM short slowpath.
     */
    test_am_send_recv(8, 8, 0, 0, UCP_OP_ATTR_FLAG_NO_IMM_CMPL);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_reply_always)


class test_ucp_am_nbx_send_copy_header : public test_ucp_am_nbx {
public:
    static void get_test_variants_reply(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_AM, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_AM, UCP_AM_SEND_FLAG_REPLY,
                               "reply");
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, get_test_variants_reply, 0);
        add_variant_values(variants, get_test_variants_reply, 1, "proto");
    }

protected:
    void test_copy_header_on_pending(size_t header_size, size_t data_size,
                                     bool rndv = false)
    {
        ucs_status_ptr_t rndv_pending_sptr = NULL;
        const unsigned flags = UCP_AM_SEND_FLAG_COPY_HEADER | get_send_flag();
        mem_buffer sbuf(data_size, tx_memtype());
        ucs_status_ptr_t pending_sptr;

        UCS_TEST_MESSAGE << "header length " << header_size << " data length "
                         << data_size << (rndv ? " RNDV" : "");

        sbuf.pattern_fill(SEED);
        m_hdr_copy.resize(header_size);
        ucs::fill_random(m_hdr_copy);
        ucp::data_type_desc_t sdt_desc(m_dt, sbuf.ptr(), data_size);
        m_hdr = m_hdr_copy;
        reset_counters();

        set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_cb, this);
        /**
         * For RNDV we use 8 byte length to fill the SQ
         * so we will not get IN_PROGRESS status from fill_sq.
         *
         * For non-RNDV, we cannot use 8 byte length,
         * because we actually want to test two cases:
         * - Get a pending request that did not yet send the first fragment.
         * - Get a pending request that sent the first fragment, but
         *   not completed sending all fragments.
         * The exact scenario depends on transport-specific properties,
         * such as tx queue length and fragment size.
         */
        size_t fillq_data_size = rndv ? 8 : data_size;
        mem_buffer fillq_sbuf(fillq_data_size, tx_memtype());
        fillq_sbuf.pattern_fill(SEED);
        ucp::data_type_desc_t fillq_sdt_desc(m_dt, fillq_sbuf.ptr(),
                                             fillq_data_size);
        pending_sptr = fill_sq(fillq_sdt_desc, header_size,
                               flags | UCP_AM_SEND_FLAG_EAGER);
        if (pending_sptr == NULL) {
            UCS_TEST_SKIP_R("Failed to get pending request");
        }

        /**
         * When testing RNDV, need to submit another request
         * with the actual message size.
         */
        if (rndv) {
            rndv_pending_sptr = send_am(sdt_desc, flags | UCP_AM_SEND_FLAG_RNDV,
                                        m_hdr_copy.data(), header_size);
        }

        ucs::fill_random(m_hdr_copy);
        while (progress());
        wait_receives();
        request_wait(pending_sptr);
        if (rndv) {
            wait_receives();
            request_wait(rndv_pending_sptr);
        }
        EXPECT_EQ(m_send_counter, m_recv_counter);
    }

    static ucs_status_t
    am_data_cb(void *arg, const void *header, size_t header_length, void *data,
               size_t length, const ucp_am_recv_param_t *param)
    {
        test_ucp_am_nbx_send_copy_header *self =
                static_cast<test_ucp_am_nbx_send_copy_header*>(arg);

        return self->am_data_handler(header, header_length, data, length,
                                     param);
    }


    static void am_data_recv_cb(void *request, ucs_status_t status,
                                size_t length, void *user_data)
    {
        test_ucp_am_nbx_send_copy_header *self =
                static_cast<test_ucp_am_nbx_send_copy_header*>(user_data);

        EXPECT_UCS_OK(status);

        self->am_recv_check_data(length);
    }

    virtual unsigned get_send_flag() const
    {
        return get_variant_value(0);
    }

private:
    ucs_status_ptr_t fill_sq(ucp::data_type_desc_t &sdt_desc,
                             size_t header_size = 0ul, unsigned flags = 0,
                             const ucp_mem_h memh = NULL)
    {
        const auto timeout = 5;
        ucs_status_ptr_t pending_sptr;

        // Warmup for wireup connection
        pending_sptr = send_am(sdt_desc, get_send_flag() | flags,
                               m_hdr_copy.data(), header_size, memh);
        wait_receives();
        request_wait(pending_sptr);
        pending_sptr = NULL;

        const ucs_time_t deadline = ucs::get_deadline(timeout);
        while ((ucs_get_time() < deadline) && (pending_sptr == NULL)) {
            pending_sptr = send_am(sdt_desc, get_send_flag() | flags,
                                   m_hdr_copy.data(), header_size, memh);
        }

        return pending_sptr;
    }

    std::string m_hdr_copy;
};

/**
 * Self transport always has resources to perform the send operation,
 * so its not returning pending request. For this reason with
 * self tl the test can't verify the copy header functionality.
 * The test is still used as a stress test for the Self transport,
 * except when running with Valgrind, because its very time consuming.
 */
UCS_TEST_SKIP_COND_P(test_ucp_am_nbx_send_copy_header, all_protos,
                     /* FIXME: Disabled due to unresolved failure - CI Hang */
                     true || (has_transport("self") && RUNNING_ON_VALGRIND),
                     "TCP_SNDBUF?=1k")
{
    const unsigned random_iterations = 20;
    const size_t max_random_hdr_len  = 64;
    const size_t max_random_data_len = 32 * UCS_KBYTE;
    size_t header_length;
    size_t data_length;

    std::vector<std::pair<unsigned, unsigned>> header_data_lengths =
            {{8, 8},
             {32, 32},
             {64, 64},
             {8, fragment_size() / 2},
             {8, fragment_size()},
             {8, fragment_size() * 2},
             {max_am_hdr(), fragment_size()}};

    for (unsigned i = 0; i < random_iterations; i++) {
        header_length = 1 + (ucs::rand() % max_random_hdr_len);
        data_length   = ucs::rand() % max_random_data_len;
        header_data_lengths.push_back(
                std::make_pair(header_length, data_length));
    }

    for (auto it : header_data_lengths) {
        header_length = it.first;
        data_length   = it.second;
        test_copy_header_on_pending(header_length, data_length);
        test_copy_header_on_pending(header_length, data_length, true);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_send_copy_header)


class test_ucp_am_nbx_send_flag : public test_ucp_am_nbx {
public:
    virtual ucs_status_t
    am_data_handler(const void *header, size_t header_length, void *data,
                    size_t length, const ucp_am_recv_param_t *rx_param)
    {
        EXPECT_FALSE(rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV);

        return test_ucp_am_nbx::am_data_handler(header, header_length, data,
                                                length, rx_param);
    }
};

UCS_TEST_P(test_ucp_am_nbx_send_flag, eager, "RNDV_THRESH=128")
{
    test_am_send_recv(256, 0, UCP_AM_SEND_FLAG_EAGER);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_send_flag)


class test_ucp_am_nbx_reply : public test_ucp_am_nbx {
public:
    static void get_test_variants(variant_vec_t &variants)
    {
        add_variant_values(variants, test_ucp_am_nbx::get_test_variants, 0);
        add_variant_values(variants, test_ucp_am_nbx::get_test_variants,
                           UCP_AM_SEND_FLAG_REPLY, "reply");
    }

protected:
    virtual unsigned get_send_flag() const
    {
        return get_variant_value(1);
    }
};


class test_ucp_am_nbx_prereg : public test_ucp_am_nbx {
public:
    static void get_test_variants(variant_vec_t &variants)
    {
        add_variant_values(variants, test_ucp_am_nbx::get_test_variants, 0);
        add_variant_values(variants, test_ucp_am_nbx::get_test_variants, 1,
                           "prereg");
    }

protected:
    virtual bool prereg() const
    {
        return get_variant_value(1);
    }
};


class test_ucp_am_nbx_closed_ep : public test_ucp_am_nbx_reply {
public:
    test_ucp_am_nbx_closed_ep()
    {
        modify_config("RESOLVE_REMOTE_EP_ID", "auto");
    }

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

    void test_recv_on_closed_ep(size_t size, bool poke_rx_progress = false,
                                bool rx_expected = false)
    {
        skip_loopback();
        test_am_send_recv(0, 0); // warmup wireup

        reset_counters();
        std::vector<char> sbuf(size, 'd');
        ucp::data_type_desc_t sdt_desc(m_dt, &sbuf[0], size);

        set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_rx_check_cb, this);

        ucs_status_ptr_t sreq = send_am(sdt_desc, get_send_flag());

        sender().progress();
        if (poke_rx_progress) {
            receiver().progress();
            if (m_send_counter == m_recv_counter) {
                request_wait(sreq);
                UCS_TEST_SKIP_R("received all AMs before ep closed");
            }
        }

        void *close_req     = receiver().disconnect_nb();
        ucs_time_t deadline = ucs::get_deadline(10);
        while (!is_request_completed(close_req) &&
               (ucs_get_time() < deadline)) {
            progress();
        };

        receiver().close_ep_req_free(close_req);

        if (rx_expected) {
            request_wait(sreq);
            wait_receives();
            EXPECT_EQ(m_recv_counter, m_send_counter);
        } else {
            // Send request may complete with error
            // (rndv should complete with EP_TIMEOUT)
            scoped_log_handler wrap_err(wrap_errors_logger);
            request_wait(sreq);
            EXPECT_LT(m_recv_counter, m_send_counter);
        }
    }
};


UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_short_am_on_closed_ep, "RNDV_THRESH=inf")
{
    // Single fragment message sent:
    // - without REPLY flag is expected to be received even if remote side
    //   closes its ep.
    // - with REPLY flag is expected to be dropped on the receiver side, when
    //   its ep is closed.
    test_recv_on_closed_ep(8, false,
                           !(get_send_flag() & UCP_AM_SEND_FLAG_REPLY));
}

// All the following type of AM messages are expected to be dropped on the
// receiver side, when its ep is closed
UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_long_am_on_closed_ep, "RNDV_THRESH=inf")
{
    test_recv_on_closed_ep(64 * UCS_KBYTE, true);
}

UCS_TEST_P(test_ucp_am_nbx_closed_ep, rx_rts_am_on_closed_ep, "RNDV_THRESH=32K")
{
    test_recv_on_closed_ep(64 * UCS_KBYTE);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_closed_ep)


class test_ucp_am_nbx_eager_memtype : public test_ucp_am_nbx_prereg {
public:
    void init()
    {
        modify_config("RNDV_THRESH", "inf");
        test_ucp_am_nbx::init();
    }

    static void get_test_variants(variant_vec_t &variants)
    {
        add_variant_memtypes(variants, base_test_generator,
                             std::numeric_limits<uint64_t>::max());
    }

private:
    static void base_test_generator(variant_vec_t &variants)
    {
        // 1. Do not instantiate test case if no GPU memtypes supported.
        // 2. Do not exclude host memory type, because this generator is used by
        //    test_ucp_am_nbx_rndv_memtype class to generate combinations like
        //    host<->cuda, cuda-managed<->host, etc.
        if (!mem_buffer::is_gpu_supported()) {
            return;
        }

        add_variant_memtypes(variants,
                             test_ucp_am_nbx_prereg::get_test_variants);
    }

    virtual ucs_memory_type_t tx_memtype() const
    {
        return static_cast<ucs_memory_type_t>(get_variant_value(2));
    }

    virtual ucs_memory_type_t rx_memtype() const
    {
        return static_cast<ucs_memory_type_t>(get_variant_value(3));
    }
};

UCS_TEST_P(test_ucp_am_nbx_eager_memtype, basic)
{
    test_am_send_recv_memtype(16 * UCS_KBYTE);
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
        EXPECT_LT(m_recv_counter, m_send_counter);
        EXPECT_TRUE(rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA);

        m_data_ptr = data;

        check_header(header, header_length);
        mem_buffer::pattern_check(data, length, SEED);
        m_recv_counter++;

        return UCS_INPROGRESS;
    }

    void test_data_release(size_t size)
    {
        size_t hdr_size = ucs_min(max_am_hdr(), 8);
        test_am_send_recv(size, 0, 0, UCP_AM_FLAG_PERSISTENT_DATA);
        ucp_am_data_release(receiver().worker(), m_data_ptr);

        test_am_send_recv(size, hdr_size, 0, UCP_AM_FLAG_PERSISTENT_DATA);
        ucp_am_data_release(receiver().worker(), m_data_ptr);
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

class test_ucp_am_nbx_align : public test_ucp_am_nbx_reply {
public:
    test_ucp_am_nbx_align()
    {
        m_alignment = pow(2, ucs::rand() % 13);
    }

    virtual ucp_worker_params_t get_worker_params()
    {
        ucp_worker_params_t params = ucp_test::get_worker_params();
        params.field_mask         |= UCP_WORKER_PARAM_FIELD_AM_ALIGNMENT;
        params.am_alignment        = m_alignment;
        return params;
    }

    virtual ucs_status_t
    am_data_handler(const void *header, size_t header_length, void *data,
                    size_t length, const ucp_am_recv_param_t *rx_param)
    {
        test_ucp_am_nbx::am_data_handler(header, header_length, data, length,
                                         rx_param);

        if (rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
            EXPECT_EQ(0u, (uintptr_t)data % m_alignment)
                      << " data ptr " << data;
        }

        return UCS_OK;
    }

private:
    size_t m_alignment;
};

UCS_TEST_P(test_ucp_am_nbx_align, basic)
{
    test_am_send_recv(fragment_size() / 2, 0, 0, UCP_AM_FLAG_PERSISTENT_DATA);
}

UCS_TEST_P(test_ucp_am_nbx_align, multi)
{
    test_am_send_recv(fragment_size() * 5, 0, 0, UCP_AM_FLAG_PERSISTENT_DATA);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_align)


class test_ucp_am_nbx_seg_size : public test_ucp_am_nbx_reply {
public:
    test_ucp_am_nbx_seg_size() : m_size(0ul)
    {
        modify_config("ADDRESS_VERSION", "v2");
        modify_config("RNDV_THRESH", "inf");
    }

    void init()
    {
        m_size = ucs_max(UCS_KBYTE, ucs::rand() % (64 * UCS_KBYTE));
        std::string str_size = ucs::to_string(m_size);

        test_ucp_am_nbx::init();

        // Create new sender() with different segment size
        modify_config("IB_SEG_SIZE", str_size, IGNORE_IF_NOT_EXIST);
        modify_config("MM_SEG_SIZE", str_size, IGNORE_IF_NOT_EXIST);
        modify_config("SCOPY_SEG_SIZE", str_size, IGNORE_IF_NOT_EXIST);
        modify_config("TCP_SEG_SIZE", str_size, IGNORE_IF_NOT_EXIST);

        entity *ent = create_entity(true);
        ent->connect(&receiver(), get_ep_params());
    }

protected:
    size_t seg_size()
    {
        return m_size;
    }

    void test_am_different_seg_sizes(size_t data_size)
    {
        UCS_TEST_MESSAGE << "seg size " << m_size << " data size " << data_size;
        test_am_send_recv(data_size);
    }

private:
    size_t m_size;
};

UCS_TEST_SKIP_COND_P(test_ucp_am_nbx_seg_size, single, has_transport("self"))
{
    test_am_different_seg_sizes(seg_size() / 2);
}

UCS_TEST_SKIP_COND_P(test_ucp_am_nbx_seg_size, multi, has_transport("self"))
{
    test_am_different_seg_sizes(seg_size() * 2);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_seg_size)


class test_ucp_am_nbx_dts : public test_ucp_am_nbx_reply {
public:
    virtual ucp_ep_params_t get_ep_params()
    {
        ucp_ep_params_t ep_params = test_ucp_am_nbx::get_ep_params();

        ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.err_mode    = get_err_mode();
        return ep_params;
    }

    static void get_test_variants(variant_vec_t &variants)
    {
        add_variant_values(variants, test_ucp_am_nbx_reply::get_test_variants,
                           UCP_ERR_HANDLING_MODE_NONE);
        add_variant_values(variants, test_ucp_am_nbx_reply::get_test_variants,
                           UCP_ERR_HANDLING_MODE_PEER, "errh");
    }

    virtual ucs_status_t
    am_data_handler(const void *header, size_t header_length, void *data,
                    size_t length, const ucp_am_recv_param_t *rx_param)
    {
        EXPECT_FALSE(rx_param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV);

        return test_ucp_am_nbx::am_data_handler(header, header_length, data,
                                                length, rx_param);
    }

private:
    ucp_err_handling_mode_t get_err_mode() const
    {
        return static_cast<ucp_err_handling_mode_t>(get_variant_value(2));
    }
};

/* Skip tests for ud_v and ud_x because of unstable reproducible failures during
 * roce on worker CI jobs. The test fails with invalid am_bcopy length. */
UCS_TEST_SKIP_COND_P(test_ucp_am_nbx_dts, short_bcopy_send,
                     is_proto_enabled() && has_any_transport({"ud_v", "ud_x"}),
                     "ZCOPY_THRESH=-1", "RNDV_THRESH=-1")
{
    test_datatypes([&]() {
        test_am(1);
        test_am(4 * UCS_KBYTE);
        test_am(64 * UCS_KBYTE);
    });
}

UCS_TEST_SKIP_COND_P(test_ucp_am_nbx_dts, zcopy_send,
                     is_proto_enabled() && has_any_transport({"ud_v", "ud_x"}),
                     "ZCOPY_THRESH=1", "RNDV_THRESH=-1")
{
    skip_no_am_lane_caps(UCT_IFACE_FLAG_AM_ZCOPY, "am_zcopy is not supported");
    test_datatypes([&]() {
        test_am(4 * UCS_KBYTE);
        test_am(64 * UCS_KBYTE);
    });
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_dts)


class test_ucp_am_nbx_rndv : public test_ucp_am_nbx_prereg {
public:
    struct am_cb_args {
        test_ucp_am_nbx_rndv *self;
        void                 **desc;
    };

    test_ucp_am_nbx_rndv()
    {
        m_status             = UCS_OK;
        m_am_recv_cb_invoked = false;
        modify_config("RNDV_THRESH", std::to_string(RNDV_THRESH));
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

        EXPECT_LT(self->m_recv_counter, self->m_send_counter);
        self->m_recv_counter++;

        return self->m_status;
    }

    static ucs_status_t am_data_deferred_rndv_cb(void *arg, const void *header,
                                                 size_t header_length,
                                                 void *data, size_t length,
                                                 const ucp_am_recv_param_t *param)
    {
        EXPECT_TRUE(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV);

        struct am_cb_args *args    = reinterpret_cast<am_cb_args*>(arg);
        test_ucp_am_nbx_rndv *self = args->self;
        void **data_desc_p         = args->desc;

        *data_desc_p               = data;
        self->m_am_recv_cb_invoked = true;

        /* Return UCS_INPROGRESS to defer handling of RNDV data */
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
        self->m_recv_counter++;

        /* return UCS_OK without calling ucp_am_recv_data_nbx()
         * to drop the message */
        return UCS_OK;
    }

    void test_am_send_deferred_recv(size_t size)
    {
        void *data_desc = NULL;
        ucp_mem_h memh  = NULL;

        mem_buffer sbuf(size, tx_memtype());
        sbuf.pattern_fill(SEED);

        struct am_cb_args args = { this,  &data_desc };
        set_am_data_handler(receiver(), TEST_AM_NBX_ID,
                            am_data_deferred_rndv_cb, &args, 0);

        if (prereg()) {
            memh = sender().mem_map(sbuf.ptr(), size);
        }

        ucp::data_type_desc_t sdt_desc(m_dt, sbuf.ptr(), size);
        ucs_status_ptr_t sptr = send_am(sdt_desc, get_send_flag(), NULL, 0,
                                        memh);

        /* Wait for AM receive callback to be invoked */
        wait_for_flag(&m_am_recv_cb_invoked);
        EXPECT_TRUE(m_am_recv_cb_invoked);

        /* Handle RNDV desc from AM receive callback */
        ucs_status_t status = am_data_rndv_handler(data_desc, size);
        ASSERT_TRUE((status == UCS_OK) || (status == UCS_INPROGRESS));
        wait_receives();
        EXPECT_EQ(m_recv_counter, m_send_counter);

        request_wait(sptr);

        if (prereg()) {
            sender().mem_unmap(memh);
        }
    }

    size_t get_rndv_frag_size(ucs_memory_type_t mem_type)
    {
        const auto *cfg = &sender().worker()->context->config.ext;

        return cfg->rndv_frag_size[mem_type];
    }

protected:
    static constexpr unsigned RNDV_THRESH = 128;
    ucs_status_t m_status;
    bool m_am_recv_cb_invoked;
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

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_zero_send_deferred_recv, "RNDV_THRESH=0")
{
    test_am_send_deferred_recv(0);
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
    ucs_status_ptr_t sptr = update_counter_and_send_am(NULL, 0ul, &data,
                                                       sizeof(data),
                                                       TEST_AM_NBX_ID, &param);

    wait_receives();

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
        reset_counters();
        m_status = statuses[i];

        ucs_status_ptr_t sptr = update_counter_and_send_am(NULL, 0ul,
                                                           sbuf.data(),
                                                           sbuf.size(),
                                                           TEST_AM_NBX_ID,
                                                           &param);

        EXPECT_EQ(m_status, request_wait(sptr));
        EXPECT_EQ(m_recv_counter, m_send_counter);
    }
}

UCS_TEST_P(test_ucp_am_nbx_rndv, deferred_reject_rndv)
{
    skip_loopback();

    void *data_desc = NULL;
    std::vector<char> sbuf(10000, 0);
    ucp_request_param_t param;

    param.op_attr_mask = 0ul;

    struct am_cb_args args = { this,  &data_desc };
    set_am_data_handler(receiver(), TEST_AM_NBX_ID, am_data_deferred_rndv_cb,
                        &args);

    ucs_status_ptr_t sptr = ucp_am_send_nbx(sender().ep(), TEST_AM_NBX_ID,
                                            NULL, 0ul, sbuf.data(),
                                            sbuf.size(), &param);

    wait_for_flag(&data_desc);
    EXPECT_TRUE(data_desc != NULL);

    ucp_am_data_release(receiver().worker(), data_desc);
    EXPECT_EQ(UCS_OK, request_wait(sptr));
}

UCS_TEST_P(test_ucp_am_nbx_rndv, dts, "RNDV_THRESH=256")
{
    test_datatypes([&]() { test_am_send_recv(64 * UCS_KBYTE); });
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_am_zcopy, "ZCOPY_THRESH=256",
           "RNDV_SCHEME=am")
{
    test_am_send_recv(256);
}

UCS_TEST_P(test_ucp_am_nbx_rndv, rndv_am_bcopy, "RNDV_SCHEME=am")
{
    test_am_send_recv(RNDV_THRESH);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_am_nbx_rndv);

class test_ucp_am_nbx_rndv_memtype : public test_ucp_am_nbx_rndv {
public:
    static void get_test_variants(variant_vec_t &variants)
    {
        // Test will not be instantiated if no GPU memtypes supported, because
        // of the check for supported memory types in
        // test_ucp_am_nbx_eager_memtype::get_test_variants
        return test_ucp_am_nbx_eager_memtype::get_test_variants(variants);
    }

    void init() override
    {
        test_ucp_am_nbx::init();
    }

private:
    unsigned get_send_flag() const override
    {
        return test_ucp_am_nbx_rndv::get_send_flag() | UCP_AM_SEND_FLAG_RNDV;
    }

    ucs_memory_type_t tx_memtype() const override
    {
        return static_cast<ucs_memory_type_t>(get_variant_value(2));
    }

    ucs_memory_type_t rx_memtype() const override
    {
        return static_cast<ucs_memory_type_t>(get_variant_value(3));
    }
};

UCS_TEST_P(test_ucp_am_nbx_rndv_memtype, rndv)
{
    const size_t rndv_frag_size     = get_rndv_frag_size(UCS_MEMORY_TYPE_HOST);
    const std::vector<size_t> sizes = {1, 64 * UCS_KBYTE, rndv_frag_size - 1,
                                       rndv_frag_size + 1};

    for (size_t size : sizes) {
        test_am_send_recv_memtype(size);
    }
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_am_nbx_rndv_memtype);


class test_ucp_am_nbx_rndv_memtype_disable_zcopy :
        public test_ucp_am_nbx_rndv_memtype {
protected:
    void disable_rndv_zcopy_config(entity &e, uint64_t zcopy_caps)
    {
        ucp_ep_h ep             = e.ep();
        ucp_ep_config_t *config = ucp_ep_config(ep);

        if (zcopy_caps & UCT_IFACE_FLAG_PUT_ZCOPY) {
            ucp_ep_config_rndv_zcopy_commit(0, &config->rndv.put_zcopy);
        }

        if (zcopy_caps & UCT_IFACE_FLAG_GET_ZCOPY) {
            ucp_ep_config_rndv_zcopy_commit(0, &config->rndv.get_zcopy);
        }
    }

    void test_disabled_rndv_zcopy(uint64_t zcopy_caps)
    {
        disable_rndv_zcopy_config(sender(), zcopy_caps);
        disable_rndv_zcopy_config(receiver(), zcopy_caps);

        test_am_send_recv(64 * UCS_KBYTE, 8, 0);
    }
};

UCS_TEST_P(test_ucp_am_nbx_rndv_memtype_disable_zcopy, rndv_disable_put_zcopy)
{
    test_disabled_rndv_zcopy(UCT_IFACE_FLAG_PUT_ZCOPY);
}

UCS_TEST_P(test_ucp_am_nbx_rndv_memtype_disable_zcopy, rndv_disable_get_zcopy)
{
    test_disabled_rndv_zcopy(UCT_IFACE_FLAG_GET_ZCOPY);
}

UCS_TEST_P(test_ucp_am_nbx_rndv_memtype_disable_zcopy,
           rndv_disable_put_and_get_zcopy)
{
    test_disabled_rndv_zcopy(UCT_IFACE_FLAG_PUT_ZCOPY |
                             UCT_IFACE_FLAG_GET_ZCOPY);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_am_nbx_rndv_memtype_disable_zcopy);


#ifdef ENABLE_STATS
class test_ucp_am_nbx_rndv_ppln : public test_ucp_am_nbx_rndv {
public:
    test_ucp_am_nbx_rndv_ppln() : m_mem_type(UCS_MEMORY_TYPE_HOST) {}

    void init() override
    {
        if (!is_proto_enabled()) {
            UCS_TEST_SKIP_R("proto v1");
        }
        stats_activate();
        modify_config("RNDV_THRESH", "128");
        modify_config("RNDV_SCHEME", "put_ppln");
        modify_config("RNDV_PIPELINE_SHM_ENABLE", "n");
        /* FIXME: Advertise error handling support for RNDV PPLN protocol.
         * Remove this once invalidation workflow is implemented. */
        modify_config("RNDV_PIPELINE_ERROR_HANDLING", "y");
        test_ucp_am_nbx::init();
    }

    void cleanup() override
    {
        test_ucp_am_nbx::cleanup();
        stats_restore();
    }

    static void get_test_variants(variant_vec_t &variants)
    {
        if (!mem_buffer::is_gpu_supported()) {
            return;
        }

        add_variant_values(variants, test_ucp_am_base::get_test_variants,
                           UCP_ERR_HANDLING_MODE_NONE);
        add_variant_values(variants, test_ucp_am_base::get_test_variants,
                           UCP_ERR_HANDLING_MODE_PEER, "errh");
    }

protected:
    virtual ucp_ep_params_t get_ep_params() override
    {
        ucp_ep_params_t ep_params = test_ucp_am_nbx_rndv::get_ep_params();
        ep_params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.err_mode        = get_err_mode();
        return ep_params;
    }

    void test_ppln_send(ucs_memory_type_t mem_type, size_t num_frags,
                        uint64_t stats_cntr_value)
    {
        if (!sender().is_rndv_put_ppln_supported()) {
            UCS_TEST_SKIP_R("RNDV pipeline is not supported");
        }

        const size_t rndv_frag_size = get_rndv_frag_size(mem_type);
        test_am_send_recv(rndv_frag_size * num_frags);

        check_stats(sender(), UCP_WORKER_STAT_RNDV_PUT_MTYPE_ZCOPY,
                    stats_cntr_value);
        check_stats(receiver(), UCP_WORKER_STAT_RNDV_RTR_MTYPE, stats_cntr_value);
    }

    void set_mem_type(ucs_memory_type_t mem_type) {
        m_mem_type = mem_type;
    }

private:
    ucs_memory_type_t tx_memtype() const override
    {
        return m_mem_type;
    }

    ucs_memory_type_t rx_memtype() const override
    {
        return m_mem_type;
    }

    void check_stats(entity &e, uint64_t cntr, uint64_t exp_value)
    {
        auto stats_node = e.worker()->stats;
        auto value      = UCS_STATS_GET_COUNTER(stats_node, cntr);

        EXPECT_EQ(exp_value, value) << "counter is "
                                    << stats_node->cls->counter_names[cntr];
    }

    ucp_err_handling_mode_t get_err_mode() const
    {
        return static_cast<ucp_err_handling_mode_t>(get_variant_value(1));
    }

    ucs_memory_type_t m_mem_type;
};

UCS_TEST_P(test_ucp_am_nbx_rndv_ppln, host_buff_cuda_frag,
           "RNDV_FRAG_MEM_TYPE=cuda")
{
    const size_t num_frags = 2;

    test_ppln_send(UCS_MEMORY_TYPE_CUDA, num_frags, num_frags);
}

UCS_TEST_P(test_ucp_am_nbx_rndv_ppln, host_buff_host_frag,
           "RNDV_FRAG_MEM_TYPE=host")
{
    // Host memory should not be pipelined thru host staging buffers
    test_ppln_send(UCS_MEMORY_TYPE_HOST, 2, 0);
}

UCS_TEST_P(test_ucp_am_nbx_rndv_ppln, cuda_buff_cuda_frag,
           "RNDV_FRAG_MEM_TYPE=cuda")
{
    const size_t num_frags = 2;

    set_mem_type(UCS_MEMORY_TYPE_CUDA);
    test_ppln_send(UCS_MEMORY_TYPE_CUDA, num_frags, num_frags);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_am_nbx_rndv_ppln);

#endif
