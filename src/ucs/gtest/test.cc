/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test.h"

extern "C" {
#include <ucs/config/parser.h>
#include <ucs/config/global_opts.h>
#include <ucs/sys/sys.h>
}

namespace ucs {

unsigned test_base::m_total_warnings = 0;

test_base::test_base() : m_state(NEW),
                m_initialized(false),
                m_num_valgrind_errors_before(0),
                m_num_warnings_before(0)
{
    push_config();
}

test_base::~test_base() {
    pop_config();
    ucs_assertv_always(m_state == FINISHED ||
                       m_state == SKIPPED ||
                       m_state == ABORTED,
                       "state=%d", m_state);
}

void test_base::set_config(const std::string& config_str)
{
    std::string::size_type pos = config_str.find("=");
    if (pos == std::string::npos) {
        modify_config(config_str, "");
    } else {
        modify_config(config_str.substr(0, pos), config_str.substr(pos + 1));
    }
}

void test_base::modify_config(const std::string& name, const std::string& value)
{
    ucs_status_t status = ucs_global_opts_set_value(name.c_str(), value.c_str());
    if (status != UCS_OK) {
        GTEST_FAIL() << "Invalid UCS configuration for " << name << " : "
                        << value << ", error message: "
                        << ucs_status_string(status) << "(" << status << ")";
    }
}

void test_base::push_config()
{
    m_config_stack.push_back(ucs_global_opts_t());
    ucs_global_opts_clone(&m_config_stack.back());
}

void test_base::pop_config()
{
    ucs_global_opts_release();
    ucs_global_opts = m_config_stack.back();
    m_config_stack.pop_back();
}

ucs_log_func_rc_t
test_base::log_handler(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level, const char *prefix, const char *message,
                       va_list ap)
{
    if (level == UCS_LOG_LEVEL_WARN) {
        ++m_total_warnings;
    }
    return UCS_LOG_FUNC_RC_CONTINUE;
}

void test_base::SetUpProxy() {
    ucs_assert(m_state == NEW);
    m_num_valgrind_errors_before = VALGRIND_COUNT_ERRORS;
    m_num_warnings_before        = m_total_warnings;

    ucs_log_push_handler(log_handler);

    try {
        init();
        m_initialized = true;
        m_state = RUNNING;
    } catch (test_skip_exception& e) {
        skipped(e);
    } catch (test_abort_exception&) {
        m_state = ABORTED;
    }
}

void test_base::TearDownProxy() {
    ucs_assertv_always(m_state == FINISHED ||
                       m_state == SKIPPED ||
                       m_state == ABORTED,
                       "state=%d", m_state);


    if (m_initialized) {
        cleanup();
    }

    ucs_log_pop_handler();

    int num_valgrind_errors = VALGRIND_COUNT_ERRORS - m_num_valgrind_errors_before;
    if (num_valgrind_errors > 0) {
        ADD_FAILURE() << "Got " << num_valgrind_errors << " valgrind errors during the test";
    }
    int num_warnings = m_total_warnings - m_num_warnings_before;
    if (num_warnings > 0) {
        ADD_FAILURE() << "Got " << num_warnings << " warnings during the test";
    }
}

void test_base::TestBodyProxy() {
    if (m_state == RUNNING) {
        try {
            test_body();
            m_state = FINISHED;
        } catch (test_skip_exception& e) {
            skipped(e);
        } catch (test_abort_exception&) {
            m_state = ABORTED;
        }
    } else if (m_state == SKIPPED) {
    } else if (m_state == ABORTED) {
    }
}

void test_base::skipped(const test_skip_exception& e) {
    std::string reason = e.what();
    if (reason.empty()) {
        detail::message_stream("SKIP");
    } else {
        detail::message_stream("SKIP") << "(" << reason << ")";
    }
    m_state = SKIPPED;
}

void test_base::init() {
}

void test_base::cleanup() {
}

}
