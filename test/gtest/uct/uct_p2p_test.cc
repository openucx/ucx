/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"
extern "C" {
#include <ucs/time/time.h>
}


int             uct_p2p_test::log_data_count = 0;
ucs_log_level_t uct_p2p_test::orig_log_level;


std::string uct_p2p_test::p2p_resource::name() const {
    std::stringstream ss;
    ss << resource::name();
    if (loopback) {
        ss << "/loopback";
    }
    return ss.str();
}

std::vector<const resource*> uct_p2p_test::enum_resources(const std::string& tl_name)
{
    static std::vector<p2p_resource> all_resources;

    if (all_resources.empty()) {
        std::vector<const resource*> r = uct_test::enum_resources("");
        for (std::vector<const resource*>::iterator iter = r.begin(); iter != r.end(); ++iter) {
            p2p_resource res;
            res.md_name    = (*iter)->md_name;
            res.local_cpus = (*iter)->local_cpus;
            res.tl_name    = (*iter)->tl_name;
            res.dev_name   = (*iter)->dev_name;
            res.dev_type   = (*iter)->dev_type;

            if (UCT_DEVICE_TYPE_SELF != res.dev_type) {
                res.loopback = false;
                all_resources.push_back(res);
            }

            res.loopback = true;
            all_resources.push_back(res);
        }
    }

    return filter_resources(all_resources, tl_name);
}

uct_p2p_test::uct_p2p_test(size_t rx_headroom,
                           uct_error_handler_t err_handler) :
    m_rx_headroom(rx_headroom),
    m_err_handler(err_handler),
    m_completion_count(0)
{
    m_null_completion      = false;
    m_completion.self      = this;
    m_completion.uct.func  = completion_cb;
    m_completion.uct.count = 0;
}

void uct_p2p_test::init() {
    uct_test::init();

    const p2p_resource *r = dynamic_cast<const p2p_resource*>(GetParam());
    ucs_assert_always(r != NULL);

    /* Create 2 connected endpoints */
    if (r->loopback) {
        entity *e = uct_test::create_entity(m_rx_headroom, m_err_handler);
        m_entities.push_back(e);
        e->connect(0, *e, 0);
    } else {
        entity *e1 = uct_test::create_entity(m_rx_headroom, m_err_handler);
        m_entities.push_back(e1);

        entity *e2 = uct_test::create_entity(m_rx_headroom, m_err_handler);
        m_entities.push_back(e2);

        e1->connect(0, *e2, 0);
        e2->connect(0, *e1, 0);
    }

    /* Allocate completion handle and set the callback */
    m_completion_count = 0;
}

void uct_p2p_test::cleanup() {

    flush();
    uct_test::cleanup();
}

void uct_p2p_test::test_xfer(send_func_t send, size_t length, unsigned flags,
                             uct_memory_type_t mem_type) {
    UCS_TEST_SKIP;
}

ucs_log_func_rc_t
uct_p2p_test::log_handler(const char *file, unsigned line, const char *function,
                          ucs_log_level_t level, const char *message, va_list ap)
{
    if (level == UCS_LOG_LEVEL_TRACE_DATA) {
        ++log_data_count;
    }

    /* Continue to next log handler if original log level would have allowed it */
    return (level <= orig_log_level) ? UCS_LOG_FUNC_RC_CONTINUE
                                     : UCS_LOG_FUNC_RC_STOP;
}

template <typename O>
void uct_p2p_test::test_xfer_print(O& os, send_func_t send, size_t length,
                                   unsigned flags, uct_memory_type_t mem_type)
{
    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
        os << ucs::size_value(length) << " " << std::flush;
    }

    /*
     * Set our own log handler, and raise log level, to test that the transport
     * prints log messages for the transfers.
     */
    int count_before = log_data_count;
    ucs_log_push_handler(log_handler);
    orig_log_level = ucs_global_opts.log_level;
    ucs_global_opts.log_level = UCS_LOG_LEVEL_TRACE_DATA;
    bool expect_log = ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA);

    UCS_TEST_SCOPE_EXIT() {
        /* Restore logging */
        ucs_global_opts.log_level = orig_log_level;
        ucs_log_pop_handler();
    } UCS_TEST_SCOPE_EXIT_END

    test_xfer(send, length, flags, mem_type);

    if (expect_log) {
        EXPECT_GE(log_data_count - count_before, 1);
    }
}

void uct_p2p_test::test_xfer_multi(send_func_t send, size_t min_length,
                                   size_t max_length, unsigned flags)
{

    for (int mem_type = 0; mem_type < UCT_MD_MEM_TYPE_LAST; mem_type++) {
        if (!((sender().md_attr().cap.mem_type == mem_type) ||
            (sender().md_attr().cap.reg_mem_types & UCS_BIT(mem_type)))) {
            continue;
        }
        if (mem_type == UCT_MD_MEM_TYPE_CUDA) {
            if (!(flags & (TEST_UCT_FLAG_RECV_ZCOPY | TEST_UCT_FLAG_SEND_ZCOPY))) {
                continue;
            }
        }
        test_xfer_multi_mem_type(send, min_length, max_length, flags,
                                 (uct_memory_type_t) mem_type);
    }
}
void uct_p2p_test::test_xfer_multi_mem_type(send_func_t send, size_t min_length,
                                            size_t max_length, unsigned flags,
                                            uct_memory_type_t mem_type) {

    ucs::detail::message_stream ms("INFO");

    ms << "memory_type:" << uct_test::uct_mem_type_names[mem_type] << " " << std::flush;

    /* Trim at 4GB */
    max_length = ucs_min(max_length, 4ull * 1124 * 1024 * 1024);

    /* Trim at max. phys memory */
    max_length = ucs_min(max_length, ucs_get_phys_mem_size() / 8);

    /* Trim at max. shared memory */
    max_length = ucs_min(max_length, ucs_get_shmmax() * 0.8);

    /* Trim when short of available memory */
    max_length = ucs_min(max_length, ucs_get_memfree_size() / 4);

    /* For large size, slow down if needed */
    if (max_length > 1 * 1024 * 1024) {
        max_length = max_length / ucs::test_time_multiplier();
        if (RUNNING_ON_VALGRIND) {
            max_length = ucs_min(max_length, 20u * 1024 * 1024);
        }
    }

    if (max_length <= min_length) {
        UCS_TEST_SKIP;
    }

    m_null_completion = false;

    /* Run with min and max values */
    test_xfer_print(ms, send, min_length, flags, mem_type);
    test_xfer_print(ms, send, max_length, flags, mem_type);

    /*
     * Generate SQRT( log(max/min) ) random sizes
     */
    double log_min = log2(min_length + 1);
    double log_max = log2(max_length - 1);

    /* How many times to repeat */
    int repeat_count;
    repeat_count = (256 * 1024) / ((max_length + min_length) / 2);
    if (repeat_count > 1000) {
        repeat_count = 1000;
    }
    repeat_count /= ucs::test_time_multiplier();
    if (repeat_count == 0) {
        repeat_count = 1;
    }

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
        ms << repeat_count << "x{" << ucs::size_value(min_length) << ".."
           << ucs::size_value(max_length) << "} " << std::flush;
    }

    for (int i = 0; i < repeat_count; ++i) {
        double exp = (ucs::rand() * (log_max - log_min)) / RAND_MAX + log_min;
        size_t length = (ssize_t)pow(2.0, exp);
        ucs_assert(length >= min_length && length <= max_length);
        test_xfer(send, length, flags, mem_type);
    }

    /* Run a test with implicit non-blocking mode */
    m_null_completion = true;
    ms << "nocomp ";
    test_xfer_print(ms, send, (long)sqrt((min_length + 1.0) * max_length),
                    flags, mem_type);

    sender().flush();
}

void uct_p2p_test::blocking_send(send_func_t send, uct_ep_h ep,
                                 const mapped_buffer &sendbuf,
                                 const mapped_buffer &recvbuf,
                                 bool wait_for_completion)
{
    unsigned prev_comp_count = m_completion_count;

    ucs_status_t status;
    do {
        status = (this->*send)(ep, sendbuf, recvbuf);
        if (status == UCS_OK) {
            return;
        } else if (status == UCS_ERR_NO_RESOURCE) {
            progress();
        } else if (status == UCS_INPROGRESS) {
            break;
        } else {
            UCS_TEST_ABORT(ucs_status_string(status));
        }
    } while (status == UCS_ERR_NO_RESOURCE);

    /* Operation in progress, wait for completion */
    ucs_assert(status == UCS_INPROGRESS);
    if (wait_for_completion) {
        if (comp() == NULL) {
            /* implicit non-blocking mode */
            sender().flush();
        } else {
            /* explicit non-blocking mode */
            ++m_completion.uct.count;
            while (m_completion_count <= prev_comp_count) {
                progress();
            }
            EXPECT_EQ(0, m_completion.uct.count);
        }
    }
}

void uct_p2p_test::wait_for_remote() {
    sender().flush();
}

uct_test::entity& uct_p2p_test::sender() {
    return **m_entities.begin();
}

uct_ep_h uct_p2p_test::sender_ep() {
    return sender().ep(0);
}

uct_test::entity& uct_p2p_test::receiver() {
    return **(m_entities.end() - 1);
}

uct_completion_t *uct_p2p_test::comp() {
    if (m_null_completion) {
        return NULL;
    } else {
        return &m_completion.uct;
    }
}

void uct_p2p_test::completion_cb(uct_completion_t *self, ucs_status_t status) {
    completion *comp = ucs_container_of(self, completion, uct);
    ++comp->self->m_completion_count;
}
