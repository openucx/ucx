/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test.h"

#include <ucs/config/parser.h>
#include <ucs/config/global_opts.h>
#include <ucs/sys/sys.h>

namespace ucs {

pthread_mutex_t test_base::m_logger_mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned test_base::m_total_warnings = 0;
unsigned test_base::m_total_errors   = 0;
std::vector<std::string> test_base::m_errors;

test_base::test_base() :
                m_state(NEW),
                m_initialized(false),
                m_num_threads(1),
                m_num_valgrind_errors_before(0),
                m_num_errors_before(0),
                m_num_warnings_before(0)
{
    push_config();
}

test_base::~test_base() {
    while (!m_config_stack.empty()) {
        pop_config();
    }
    ucs_assertv_always(m_state == FINISHED ||
                       m_state == SKIPPED ||
                       m_state == NEW ||    /* can be skipped from a class constructor */
                       m_state == ABORTED,
                       "state=%d", m_state);
}

void test_base::set_num_threads(unsigned num_threads) {
    if (m_state != NEW) {
        GTEST_FAIL() << "Cannot modify number of threads after test is started, "
                     << "it must be done in the constructor.";
    }
    m_num_threads = num_threads;
}

unsigned test_base::num_threads() const {
    return m_num_threads;
}

void test_base::set_config(const std::string& config_str)
{
    std::string::size_type pos = config_str.find("=");
    std::string name, value;
    bool optional;

    if (pos == std::string::npos) {
        name  = config_str;
        value = "";
    } else {
        name  = config_str.substr(0, pos);
        value = config_str.substr(pos + 1);
    }

    optional = false;
    if ((name.length() > 0) && name.at(name.length() - 1) == '?') {
        name = name.substr(0, name.length() - 1);
        optional = true;
    }

    modify_config(name, value, optional);
}

void test_base::get_config(const std::string& name, std::string& value, size_t max)
{
    ucs_status_t status;

    value.resize(max, '\0');
    status = ucs_global_opts_get_value(name.c_str(),
                                       const_cast<char*>(value.c_str()),
                                       max);
    if (status != UCS_OK) {
        GTEST_FAIL() << "Invalid UCS configuration for " << name
                     << ", error message: " << ucs_status_string(status)
                     << "(" << status << ")";
    }
}

void test_base::modify_config(const std::string& name, const std::string& value,
                              bool optional)
{
    ucs_status_t status = ucs_global_opts_set_value(name.c_str(), value.c_str());
    if ((status == UCS_OK) || (optional && (status == UCS_ERR_NO_ELEM))) {
        return;
    }

    GTEST_FAIL() << "Invalid UCS configuration for " << name << " : "
                    << value << ", error message: "
                    << ucs_status_string(status) << "(" << status << ")";
}

void test_base::push_config()
{
    ucs_global_opts_t new_opts;
    /* save current options to the vector
     * it is important to keep the first original global options at the first
     * vector element to release it at the end. Otherwise, memtrack will not work
     */
    m_config_stack.push_back(ucs_global_opts);
    ucs_global_opts_clone(&new_opts);
    ucs_global_opts = new_opts;
}

void test_base::pop_config()
{
    ucs_global_opts_release();
    ucs_global_opts = m_config_stack.back();
    m_config_stack.pop_back();
}

void test_base::hide_errors()
{
    ucs_log_push_handler(hide_errors_logger);
}

void test_base::wrap_errors()
{
    ucs_log_push_handler(wrap_errors_logger);
}

void test_base::restore_errors()
{
    ucs_log_pop_handler();
}

ucs_log_func_rc_t
test_base::count_warns_logger(const char *file, unsigned line, const char *function,
                              ucs_log_level_t level, const char *message, va_list ap)
{
    pthread_mutex_lock(&m_logger_mutex);
    if (level == UCS_LOG_LEVEL_ERROR) {
        ++m_total_errors;
    } else if (level == UCS_LOG_LEVEL_WARN) {
        ++m_total_warnings;
    }
    pthread_mutex_unlock(&m_logger_mutex);
    return UCS_LOG_FUNC_RC_CONTINUE;
}

std::string test_base::format_message(const char *message, va_list ap)
{
    char buf[ucs_global_opts.log_buffer_size];
    vsnprintf(buf, ucs_global_opts.log_buffer_size, message, ap);
    return std::string(buf);
}

ucs_log_func_rc_t
test_base::hide_errors_logger(const char *file, unsigned line, const char *function,
                              ucs_log_level_t level, const char *message, va_list ap)
{
    if (level == UCS_LOG_LEVEL_ERROR) {
        pthread_mutex_lock(&m_logger_mutex);
        va_list ap2;
        va_copy(ap2, ap);
        m_errors.push_back(format_message(message, ap2));
        va_end(ap2);
        level = UCS_LOG_LEVEL_DEBUG;
        pthread_mutex_unlock(&m_logger_mutex);
    }

    ucs_log_default_handler(file, line, function, level, message, ap);
    return UCS_LOG_FUNC_RC_STOP;
}

ucs_log_func_rc_t
test_base::wrap_errors_logger(const char *file, unsigned line, const char *function,
                              ucs_log_level_t level, const char *message, va_list ap)
{
    /* Ignore warnings about empty memory pool */
    if (level == UCS_LOG_LEVEL_ERROR) {
        pthread_mutex_lock(&m_logger_mutex);
        std::string text = format_message(message, ap);
        m_errors.push_back(text);
        UCS_TEST_MESSAGE << "< " << text << " >";
        pthread_mutex_unlock(&m_logger_mutex);
        return UCS_LOG_FUNC_RC_STOP;
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

void test_base::SetUpProxy() {
    ucs_assert(m_state == NEW);
    m_num_valgrind_errors_before = VALGRIND_COUNT_ERRORS;
    m_num_warnings_before        = m_total_warnings;
    m_num_errors_before          = m_total_errors;

    m_errors.clear();
    ucs_log_push_handler(count_warns_logger);

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
    m_errors.clear();

    int num_valgrind_errors = VALGRIND_COUNT_ERRORS - m_num_valgrind_errors_before;
    if (num_valgrind_errors > 0) {
        ADD_FAILURE() << "Got " << num_valgrind_errors << " valgrind errors during the test";
    }
    int num_errors = m_total_errors - m_num_errors_before;
    if (num_errors > 0) {
        ADD_FAILURE() << "Got " << num_errors << " errors during the test";
    }
    int num_warnings = m_total_warnings - m_num_warnings_before;
    if (num_warnings > 0) {
        ADD_FAILURE() << "Got " << num_warnings << " warnings during the test";
    }
}

void test_base::run()
{
    if (num_threads() == 1) {
        test_body();
    } else {
        pthread_t threads[num_threads()];
        pthread_barrier_init(&m_barrier, NULL, num_threads());
        for (unsigned i = 0; i < num_threads(); ++i) {
            pthread_create(&threads[i], NULL, thread_func, reinterpret_cast<void*>(this));
        }
        for (unsigned i = 0; i < num_threads(); ++i) {
            void *retval;
            pthread_join(threads[i], &retval);
        }
        pthread_barrier_destroy(&m_barrier);
    }
}

void *test_base::thread_func(void *arg)
{
    test_base *self = reinterpret_cast<test_base*>(arg);
    self->barrier(); /* Let all threads start in the same time */
    self->test_body();
    return NULL;
}

void test_base::TestBodyProxy() {
    if (m_state == RUNNING) {
        try {
            run();
            m_state = FINISHED;
        } catch (test_skip_exception& e) {
            skipped(e);
        } catch (test_abort_exception&) {
            m_state = ABORTED;
        } catch (exit_exception& e) {
            if (RUNNING_ON_VALGRIND) {
                /* When running with valgrind, exec true/false instead of just
                 * exiting, to avoid warnings about memory leaks of objects
                 * allocated inside gtest run loop.
                 */
                const char *program = e.failed() ? "false" : "true";
                execlp(program, program, NULL);
            }

            /* If not running on valgrind / execp failed, use exit() */
            exit(e.failed() ? 1 : 0);
        } catch (...) {
            m_state = ABORTED;
            throw;
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

bool test_base::barrier() {
    int ret = pthread_barrier_wait(&m_barrier);
    if (ret == 0) {
        return false;
    } else if (ret == PTHREAD_BARRIER_SERIAL_THREAD) {
        return true;
    } else {
        UCS_TEST_ABORT("pthread_barrier_wait() failed");
    }

}

}
