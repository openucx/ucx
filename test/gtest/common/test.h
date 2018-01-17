/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TEST_BASE_H
#define UCS_TEST_BASE_H

#include "test_helpers.h"

#include <ucs/debug/log.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/config/parser.h>

#include <map>
#include <vector>
#include <string>

namespace ucs {

/**
 * Base class for tests
 */
class test_base {
public:
    test_base();
    virtual ~test_base();

    void set_num_threads(unsigned num_threads);
    unsigned num_threads() const;

    void get_config(const std::string& name, std::string& value,
                            size_t max);
    virtual void set_config(const std::string& config_str);
    virtual void modify_config(const std::string& name, const std::string& value,
                               bool optional = false);
    virtual void push_config();
    virtual void pop_config();

    static void hide_errors();
    static void wrap_errors();
    static void restore_errors();

protected:

    typedef enum {
        NEW, RUNNING, SKIPPED, ABORTED, FINISHED
    } state_t;

    typedef std::vector<ucs_global_opts_t> config_stack_t;

    void SetUpProxy();
    void TearDownProxy();
    void TestBodyProxy();
    static std::string format_message(const char *message, va_list ap);

    virtual void cleanup();
    virtual void init();
    bool barrier();

    virtual void test_body() = 0;

    state_t              m_state;
    bool                 m_initialized;
    unsigned             m_num_threads;
    config_stack_t       m_config_stack;
    int                  m_num_valgrind_errors_before;
    unsigned             m_num_errors_before;
    unsigned             m_num_warnings_before;

    static pthread_mutex_t          m_logger_mutex;
    static unsigned                 m_total_errors;
    static unsigned                 m_total_warnings;
    static std::vector<std::string> m_errors;

private:
    void skipped(const test_skip_exception& e);
    void run();
    static void *thread_func(void *arg);

    static ucs_log_func_rc_t
    count_warns_logger(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level, const char *message, va_list ap);

    static ucs_log_func_rc_t
    hide_errors_logger(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level, const char *message, va_list ap);

    static ucs_log_func_rc_t
    wrap_errors_logger(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level, const char *message, va_list ap);

    pthread_barrier_t    m_barrier;
};

#define UCS_TEST_BASE_IMPL \
    virtual void SetUp() { \
        test_base::SetUpProxy(); \
    } \
    \
    virtual void TearDown() { \
        test_base::TearDownProxy(); \
    } \
    virtual void TestBody() { \
        test_base::TestBodyProxy(); \
    }

/*
 * Base class from generic tests
 */
class test : public testing::Test, public test_base {
public:
    UCS_TEST_BASE_IMPL;
};

/**
 * UCT/UCP tests common storage for tests entities
 */
template <typename T>
class entities_storage {
public:
    const ucs::ptr_vector<T>& entities() const {
        return m_entities;
    }

    T& sender() {
        return *m_entities.front();
    }

    T& receiver() {
        return *m_entities.back();
    }

    T& e(size_t idx) {
        return m_entities.at(idx);
    }

    bool is_loopback() {
        return &sender() == &receiver();
    }

    void skip_loopback() {
        if (is_loopback()) {
            UCS_TEST_SKIP_R("loopback");
        }
    }

    ucs::ptr_vector<T> m_entities;
};

}

#define UCS_TEST_SET_CONFIG(_dummy, _config) \
    set_config(_config);

/*
 * Helper macro
 */
#define UCS_TEST_(test_case_name, test_name, parent_class, parent_id, num_threads, ...)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {\
     set_num_threads(num_threads); \
     UCS_PP_FOREACH(UCS_TEST_SET_CONFIG, _, __VA_ARGS__) \
  } \
 private:\
  virtual void test_body();\
  static ::testing::TestInfo* const test_info_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
};\
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ =\
    ::testing::internal::MakeAndRegisterTestInfo(\
        #test_case_name, \
        (num_threads == 1) ? #test_name : #test_name "/mt_" #num_threads, \
        "", "", \
        (parent_id), \
        parent_class::SetUpTestCase, \
        parent_class::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl<\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::test_body()


/*
 * Define test fixture with modified configuration
 */
#define UCS_TEST_F(test_fixture, test_name, ...)\
  UCS_TEST_(test_fixture, test_name, test_fixture, \
            ::testing::internal::GetTypeId<test_fixture>(), 1, __VA_ARGS__)


/*
 * Define test fixture with multiple threads
 */
#define UCS_MT_TEST_F(test_fixture, test_name, num_threads, ...)\
  UCS_TEST_(test_fixture, test_name, test_fixture, \
            ::testing::internal::GetTypeId<test_fixture>(), num_threads, __VA_ARGS__)


/*
 * Helper macro
 */
#define UCS_TEST_P_(test_case_name, test_name, num_threads, ...) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {\
       set_num_threads(num_threads); \
       UCS_PP_FOREACH(UCS_TEST_SET_CONFIG, _, __VA_ARGS__); \
    } \
    virtual void test_body(); \
   private: \
    static int AddToRegistry() { \
      ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
          GetTestCasePatternHolder<test_case_name>(\
              #test_case_name, __FILE__, __LINE__)->AddTestPattern(\
                  #test_case_name, \
                  (num_threads == 1) ? #test_name : #test_name "/mt_" #num_threads, \
                  new ::testing::internal::TestMetaFactory< \
                      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>()); \
      return 0; \
    } \
    static int gtest_registering_dummy_; \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(\
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
  }; \
  int GTEST_TEST_CLASS_NAME_(test_case_name, \
                             test_name)::gtest_registering_dummy_ = \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
  void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::test_body()


/*
 * Define parameterized test with modified configuration
 */
#define UCS_TEST_P(test_case_name, test_name, ...) \
    UCS_TEST_P_(test_case_name, test_name, 1, __VA_ARGS__)


/*
 * Define parameterized test with multiple threads
 */
#define UCS_MT_TEST_P(test_case_name, test_name, num_threads, ...) \
    UCS_TEST_P_(test_case_name, test_name, num_threads, __VA_ARGS__)

#endif
