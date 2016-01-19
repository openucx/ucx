/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <ucs/gtest/test.h>

class test_pd : public testing::TestWithParam<std::string>,
                public ucs::test_base
{
public:
    UCS_TEST_BASE_IMPL;

    static std::vector<std::string> enum_pds(const std::string& pdc_name);

    test_pd();

protected:
    virtual void init();
    virtual void cleanup();
    virtual void modify_config(const std::string& name, const std::string& value);
    void check_caps(uint64_t flags, const std::string& name);

    void test_registration();

    uct_pd_h pd() {
        return m_pd;
    }

private:
    ucs::handle<uct_pd_config_t*> m_pd_config;
    ucs::handle<uct_pd_h>         m_pd;
};

std::vector<std::string> test_pd::enum_pds(const std::string& pdc_name) {
    static std::vector<std::string> all_pds;
    std::vector<std::string> result;

    if (all_pds.empty()) {
        uct_pd_resource_desc_t *pd_resources;
        unsigned num_pd_resources;
        ucs_status_t status;

        status = uct_query_pd_resources(&pd_resources, &num_pd_resources);
        ASSERT_UCS_OK(status);

        for (unsigned i = 0; i < num_pd_resources; ++i) {
            all_pds.push_back(pd_resources[i].pd_name);
        }

        uct_release_pd_resource_list(pd_resources);
    }

    for (std::vector<std::string>::iterator iter = all_pds.begin();
                    iter != all_pds.end(); ++iter)
    {
        if (iter->substr(0, pdc_name.length()) == pdc_name) {
            result.push_back(*iter);
        }
    }
    return result;
}

test_pd::test_pd()
{
    UCS_TEST_CREATE_HANDLE(uct_pd_config_t*, m_pd_config,
                           (void (*)(uct_pd_config_t*))uct_config_release,
                           uct_pd_config_read, GetParam().c_str(), NULL, NULL);
}

void test_pd::init()
{
    ucs::test_base::init();
    UCS_TEST_CREATE_HANDLE(uct_pd_h, m_pd, uct_pd_close, uct_pd_open,
                           GetParam().c_str(), m_pd_config);
}

void test_pd::cleanup()
{
    m_pd.reset();
    ucs::test_base::cleanup();
}

void test_pd::modify_config(const std::string& name, const std::string& value)
{
    ucs_status_t status = uct_config_modify(m_pd_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        return ucs::test_base::modify_config(name, value);
    } else {
        ASSERT_UCS_OK(status);
    }
}

void test_pd::check_caps(uint64_t flags, const std::string& name)
{
    uct_pd_attr_t pd_attr;
    ucs_status_t status = uct_pd_query(pd(), &pd_attr);
    ASSERT_UCS_OK(status);
    if (!ucs_test_all_flags(pd_attr.cap.flags, flags)) {
        std::stringstream ss;
        ss << name << " is not supported by " << GetParam();
        UCS_TEST_SKIP_R(ss.str());
    }
}

UCS_TEST_P(test_pd, alloc) {
    size_t size, orig_size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    check_caps(UCT_PD_FLAG_ALLOC, "allocation");

    for (unsigned i = 0; i < 300; ++i) {
        size = orig_size = rand() % 65536;
        if (size == 0) {
            continue;
        }

        status = uct_pd_mem_alloc(pd(), &size, &address, "test", &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);
        EXPECT_GE(size, orig_size);
        EXPECT_TRUE(address != NULL);
        EXPECT_TRUE(memh != UCT_INVALID_MEM_HANDLE);

        memset(address, 0xBB, size);
        uct_pd_mem_free(pd(), memh);
    }
}

UCS_TEST_P(test_pd, reg) {
    size_t size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    check_caps(UCT_PD_FLAG_REG, "registration");

    for (unsigned i = 0; i < 300; ++i) {
        size = rand() % 65536;
        if (size == 0) {
            continue;
        }

        address = malloc(size);
        ASSERT_TRUE(address != NULL);

        memset(address, 0xBB, size);

        status = uct_pd_mem_reg(pd(), address, size, &memh);

        ASSERT_UCS_OK(status);
        ASSERT_TRUE(memh != UCT_INVALID_MEM_HANDLE);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        status = uct_pd_mem_dereg(pd(), memh);
        ASSERT_UCS_OK(status);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        free(address);
    }
}

UCS_TEST_P(test_pd, reg_perf) {
    static const unsigned count = 10000;
    ucs_status_t status;

    check_caps(UCT_PD_FLAG_REG, "registration");

    for (size_t size = 4096; size <= 4 * 1024 * 1024; size *= 2) {
        void *ptr = malloc(size);
        ASSERT_TRUE(ptr != NULL);
        memset(ptr, 0xBB, size);

        ucs_time_t start_time = ucs_get_time();
        for (unsigned i = 0; i < count; ++i) {
            uct_mem_h memh;
            status = uct_pd_mem_reg(pd(), ptr, size, &memh);
            ASSERT_UCS_OK(status);
            ASSERT_TRUE(memh != UCT_INVALID_MEM_HANDLE);

            status = uct_pd_mem_dereg(pd(), memh);
            ASSERT_UCS_OK(status);
        }
        ucs_time_t end_time = ucs_get_time();

        UCS_TEST_MESSAGE << GetParam() << ": Registration time for " <<
                        size << " bytes: " <<
                        long(ucs_time_to_nsec(end_time - start_time) / count) << " ns";

        free(ptr);
    }
}


#define UCT_PD_INSTANTIATE_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_PD_INSTANTIATE_TEST_CASE, _test_case, \
                   knem, \
                   cma, \
                   posiz, \
                   sysv, \
                   xpmem, \
                   cuda, \
                   ib, \
                   ugni \
                   )
#define _UCT_PD_INSTANTIATE_TEST_CASE(_test_case, _pdc_name) \
    INSTANTIATE_TEST_CASE_P(_pdc_name, _test_case, \
                            testing::ValuesIn(_test_case::enum_pds(#_pdc_name)));

UCT_PD_INSTANTIATE_TEST_CASE(test_pd)
