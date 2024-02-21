/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
# Copyright (C) NextSilicon Ltd. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TEST_BASE_H
#define UCS_TEST_BASE_H

/* gcc 4.3.4 compilation */
#ifndef UINT8_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS 1
#endif

#include <inttypes.h>

#include "test_helpers.h"

#include <ucs/debug/log.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/config/parser.h>

#include <map>
#include <vector>
#include <string>

namespace ucs {

namespace log {
    class scoped_handler {
    public:
        scoped_handler(ucs_log_func_t handler) {
            ucs_log_push_handler(handler);
        }
        ~scoped_handler() {
            ucs_log_pop_handler();
        }
    };

    void setup();
    void teardown();
    const std::vector<std::string>& errors();
    const std::vector<std::string>& warnings();
    std::vector<std::string>& warnings_raw();

    size_t num_errors();
    size_t num_warnings();

    std::string format_message(const char *message, va_list ap);

    ucs_log_func_rc_t
    common_logger(ucs_log_level_t log_level_to_handle, bool print,
                  std::vector<std::string> &messages_vec, size_t limit,
                  const char *file, unsigned line, const char *function,
                  ucs_log_level_t level,
                  const ucs_log_component_config_t *comp_conf,
                  const char *message, va_list ap);

    ucs_log_func_rc_t
    count_warns_logger(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level,
                       const ucs_log_component_config_t *comp_conf,
                       const char *message, va_list ap);

    ucs_log_func_rc_t
    hide_errors_logger(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level,
                       const ucs_log_component_config_t *comp_conf,
                       const char *message, va_list ap);

    ucs_log_func_rc_t
    hide_warns_logger(const char *file, unsigned line, const char *function,
                      ucs_log_level_t level,
                      const ucs_log_component_config_t *comp_conf,
                      const char *message, va_list ap);

    ucs_log_func_rc_t
    wrap_errors_logger(const char *file, unsigned line, const char *function,
                       ucs_log_level_t level,
                       const ucs_log_component_config_t *comp_conf,
                       const char *message, va_list ap);

    ucs_log_func_rc_t
    wrap_warns_logger(const char *file, unsigned line, const char *function,
                      ucs_log_level_t level,
                      const ucs_log_component_config_t *comp_conf,
                      const char *message, va_list ap);
};

/**
 * Base class for tests
 */
class test_base {
public:
    typedef enum {
        IGNORE_IF_NOT_EXIST,
        FAIL_IF_NOT_EXIST,
        SETENV_IF_NOT_EXIST,
        SKIP_IF_NOT_EXIST
    } modify_config_mode_t;

    test_base();
    virtual ~test_base();

    void set_num_threads(unsigned num_threads);
    unsigned num_threads() const;

    virtual void set_config(const std::string& config_str = "");
    virtual void modify_config(const std::string& name, const std::string& value,
                               modify_config_mode_t mode = FAIL_IF_NOT_EXIST);
    virtual void push_config();
    virtual void pop_config();

    void stats_activate();
    void stats_restore();
protected:
    typedef enum {
        NEW, RUNNING, SKIPPED, ABORTED, FINISHED
    } state_t;

    typedef std::vector<ucs_global_opts_t> config_stack_t;

    void SetUpProxy();
    void TearDownProxy();
    void TestBodyProxy();

    virtual void cleanup();
    virtual void init();
    bool barrier();

    virtual void check_skip_test() = 0;

    virtual void test_body() = 0;

    state_t                         m_state;
    bool                            m_initialized;
    unsigned                        m_num_threads;
    config_stack_t                  m_config_stack;
    ptr_vector<scoped_setenv>       m_env_stack;

private:
    void skipped(const test_skip_exception& e);
    void run();

    static void *thread_func(void *arg);

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

/*
 * Base class from generic tests with user-defined parameter
 */
template <typename T>
class test_with_param : public testing::TestWithParam<T>, public test_base {
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
        return sender(0);
    }

    T& sender(size_t idx) {
        return e(idx);
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
/* Make sure no MADV_DONTCOPY memory areas left behind when constructed.
 * Tests which use fork()/system() should inherit from this class as 1st parent,
 * to make sure its constructor is called before any other parent's.
 */
class clear_dontcopy_regions {
public:
    clear_dontcopy_regions();
};

}

#define UCS_TEST_SET_CONFIG(_dummy, _config) \
    set_config(_config);

/*
 * Helper macro
 */
#define UCS_TEST_(test_case_name, test_name, parent_id, \
                  num_threads, skip_cond, skip_reason, ...) \
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public test_case_name { \
 public: \
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() { \
     set_num_threads(num_threads); \
  } \
 protected: \
  virtual void init() { \
      UCS_PP_FOREACH(UCS_TEST_SET_CONFIG, _, __VA_ARGS__) \
	  test_case_name::init(); \
  } \
 private: \
  virtual void check_skip_test() { \
      if (skip_cond) { \
          UCS_TEST_SKIP_R(skip_reason); \
      } \
  } \
  virtual void test_body(); \
  static ::testing::TestInfo* const test_info_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
}; \
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ = \
    ::testing::internal::MakeAndRegisterTestInfo( \
        #test_case_name, \
        (num_threads == 1) ? #test_name : #test_name "/mt_" #num_threads, \
        "", "", \
        ::testing::internal::CodeLocation(__FILE__, __LINE__), \
        (parent_id), \
		test_case_name::SetUpTestCase, \
		test_case_name::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl< \
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>); \
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::test_body()


/*
 * Define test fixture with modified configuration
 */
#define UCS_TEST_F(test_fixture, test_name, ...) \
  UCS_TEST_(test_fixture, test_name, \
            ::testing::internal::GetTypeId<test_fixture>(), \
            1, 0, "", __VA_ARGS__)


/*
 * Define test fixture with modified configuration and check skip condition
 */
#define UCS_TEST_SKIP_COND_F(test_fixture, test_name, skip_cond, ...) \
  UCS_TEST_(test_fixture, test_name, \
            ::testing::internal::GetTypeId<test_fixture>(), \
            1, skip_cond, #skip_cond, __VA_ARGS__)


/*
 * Define test fixture with multiple threads
 */
#define UCS_MT_TEST_F(test_fixture, test_name, num_threads, ...) \
  UCS_TEST_(test_fixture, test_name, \
            ::testing::internal::GetTypeId<test_fixture>(), \
            num_threads, 0, "", __VA_ARGS__)


/*
 * Helper macro
 */
#define UCS_TEST_P_(test_case_name, test_name, num_threads, \
                    skip_cond, skip_reason, ...) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() { \
       set_num_threads(num_threads); \
    } \
    virtual void test_body(); \
   protected: \
    virtual void init() { \
        UCS_PP_FOREACH(UCS_TEST_SET_CONFIG, _, __VA_ARGS__) \
		test_case_name::init(); \
    } \
   private: \
    virtual void check_skip_test() { \
        if (skip_cond) { \
            UCS_TEST_SKIP_R(skip_reason); \
        } \
    } \
    static int AddToRegistry() { \
        ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
            GetTestCasePatternHolder<test_case_name>( \
                #test_case_name, ::testing::internal::CodeLocation(__FILE__, __LINE__))->AddTestPattern( \
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
    UCS_TEST_P_(test_case_name, test_name, 1, 0, "", __VA_ARGS__)


/*
 * Define parameterized test with modified configuration and check skip condition
 */
#define UCS_TEST_SKIP_COND_P(test_case_name, test_name, skip_cond, ...) \
    UCS_TEST_P_(test_case_name, test_name, 1, skip_cond, #skip_cond, __VA_ARGS__)


/*
 * Define parameterized test with multiple threads
 */
#define UCS_MT_TEST_P(test_case_name, test_name, num_threads, ...) \
    UCS_TEST_P_(test_case_name, test_name, num_threads, 0, "", __VA_ARGS__)

#endif
