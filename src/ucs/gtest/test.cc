/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "test.h"

extern "C" {
#include <ucs/config/parser.h>
#include <ucs/config/global_opts.h>
}

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

namespace ucs {

test_base::test_base() : m_state(NEW) {
}

test_base::~test_base() {
    ucs_assertv_always(m_state == FINISHED ||
                       m_state == SKIPPED ||
                       m_state == ABORTED,
                       "state=%d", m_state);
}

void test_base::set_config(void *opts, ucs_config_field_t *fields,
                           const std::string& name, const std::string& value)
{
    ucs_status_t status = ucs_config_parser_set_value(opts, fields, name.c_str(),
                                                      value.c_str());
    if (status != UCS_OK) {
        GTEST_FAIL() << "Invalid UCS configuration for " << name << " : " << value;
    }
}

void test_base::set_config(const std::string& name, const std::string& value)
{
    set_config(&ucs_global_opts, ucs_global_opts_table, name, value);
}

void test_base::push_config()
{
    m_config_stack.push_back(ucs_global_opts_t());
    ucs_config_parser_clone_opts(&ucs_global_opts, &m_config_stack.back(),
                                 ucs_global_opts_table);
}

void test_base::pop_config()
{
    ucs_config_parser_release_opts(&ucs_global_opts, ucs_global_opts_table);
    ucs_global_opts = m_config_stack.back();
}

void test_base::set_multi_config(const std::string& multi_config) {
    std::vector<std::string> tokens;
    boost::split(tokens, multi_config, boost::is_any_of("="));
    set_config(tokens[0], tokens[1]);
}

void test_base::SetUpProxy() {
    ucs_assert(m_state == NEW);
    m_num_valgrind_errors_before = VALGRIND_COUNT_ERRORS;

    try {
        init();
        m_state = RUNNING;
    } catch (test_skip_exception&) {
        detail::message_stream("SKIP");
        m_state = SKIPPED;
    } catch (test_abort_exception&) {
        m_state = ABORTED;
    }
}

void test_base::TearDownProxy() {
    ucs_assertv_always(m_state == FINISHED ||
                       m_state == SKIPPED ||
                       m_state == ABORTED,
                       "state=%d", m_state);
    cleanup();
    int num_valgrind_errors = VALGRIND_COUNT_ERRORS - m_num_valgrind_errors_before;
    if (num_valgrind_errors > 0) {
        ADD_FAILURE() << "Got " << num_valgrind_errors << " valgrind errors during the test";
    }
}

void test_base::TestBodyProxy() {
    if (m_state == RUNNING) {
        try {
            test_body();
            m_state = FINISHED;
        } catch (test_skip_exception&) {
            detail::message_stream("SKIP");
            m_state = SKIPPED;
        } catch (test_abort_exception&) {
            m_state = ABORTED;
        }
    } else if (m_state == SKIPPED) {
    } else if (m_state == ABORTED) {
    }
}

void test_base::init() {
}

void test_base::cleanup() {
}

}
