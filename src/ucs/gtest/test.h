/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_TEST_BASE_H
#define UCS_TEST_BASE_H

#include "test_helpers.h"

extern "C" {
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/config/parser.h>
}

#include <map>
#include <vector>
#include <string>

namespace ucs {

/**
 * Base class for tests
 */
class test_base {
protected:
    test_base();
    virtual ~test_base();

    virtual void set_config(const std::string& config_str);
    virtual void set_config(const std::string& name, const std::string& value);
    virtual void push_config();
    virtual void pop_config();

    /* Helpers */
    void set_config(void *opts, ucs_config_field_t *fields,
                    const std::string& name, const std::string& value);

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

    virtual void test_body() = 0;

    config_stack_t       m_config_stack;
    state_t              m_state;
    bool                 m_skip;
    int                  m_num_valgrind_errors_before;

private:
    void skipped(const test_skip_exception& e);
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

}

#define UCS_TEST_SET_CONFIG(_dummy, _config) \
    set_config(_config);

/*
 * Helper macro
 */
#define UCS_TEST_(test_case_name, test_name, parent_class, parent_id, ...)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {\
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
        #test_case_name, #test_name, "", "", \
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
            ::testing::internal::GetTypeId<test_fixture>(), __VA_ARGS__)


/*
 * Define parameterized test with modified configuration
 */
#define UCS_TEST_P(test_case_name, test_name, ...) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {\
       UCS_PP_FOREACH(UCS_TEST_SET_CONFIG, _, __VA_ARGS__); \
    } \
    virtual void test_body(); \
   private: \
    static int AddToRegistry() { \
      ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
          GetTestCasePatternHolder<test_case_name>(\
              #test_case_name, __FILE__, __LINE__)->AddTestPattern(\
                  #test_case_name, \
                  #test_name, \
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

#endif
