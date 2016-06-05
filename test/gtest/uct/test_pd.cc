/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <common/test.h>

class test_pd : public testing::TestWithParam<std::string>,
                public ucs::test_base
{
public:
    UCS_TEST_BASE_IMPL;

    static std::vector<std::string> enum_mds(const std::string& mdc_name);

    test_pd();

protected:
    virtual void init();
    virtual void cleanup();
    virtual void modify_config(const std::string& name, const std::string& value);
    void check_caps(uint64_t flags, const std::string& name);

    void test_registration();

    uct_md_h pd() {
        return m_pd;
    }

private:
    ucs::handle<uct_md_config_t*> m_md_config;
    ucs::handle<uct_md_h>         m_pd;
};

std::vector<std::string> test_pd::enum_mds(const std::string& mdc_name) {
    static std::vector<std::string> all_pds;
    std::vector<std::string> result;

    if (all_pds.empty()) {
        uct_md_resource_desc_t *md_resources;
        unsigned num_md_resources;
        ucs_status_t status;

        status = uct_query_md_resources(&md_resources, &num_md_resources);
        ASSERT_UCS_OK(status);

        for (unsigned i = 0; i < num_md_resources; ++i) {
            all_pds.push_back(md_resources[i].md_name);
        }

        uct_release_md_resource_list(md_resources);
    }

    for (std::vector<std::string>::iterator iter = all_pds.begin();
                    iter != all_pds.end(); ++iter)
    {
        if (iter->substr(0, mdc_name.length()) == mdc_name) {
            result.push_back(*iter);
        }
    }
    return result;
}

test_pd::test_pd()
{
    UCS_TEST_CREATE_HANDLE(uct_md_config_t*, m_md_config,
                           (void (*)(uct_md_config_t*))uct_config_release,
                           uct_md_config_read, GetParam().c_str(), NULL, NULL);
}

void test_pd::init()
{
    ucs::test_base::init();
    UCS_TEST_CREATE_HANDLE(uct_md_h, m_pd, uct_md_close, uct_md_open,
                           GetParam().c_str(), m_md_config);
}

void test_pd::cleanup()
{
    m_pd.reset();
    ucs::test_base::cleanup();
}

void test_pd::modify_config(const std::string& name, const std::string& value)
{
    ucs_status_t status = uct_config_modify(m_md_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        return ucs::test_base::modify_config(name, value);
    } else {
        ASSERT_UCS_OK(status);
    }
}

void test_pd::check_caps(uint64_t flags, const std::string& name)
{
    uct_md_attr_t md_attr;
    ucs_status_t status = uct_md_query(pd(), &md_attr);
    ASSERT_UCS_OK(status);
    if (!ucs_test_all_flags(md_attr.cap.flags, flags)) {
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

    check_caps(UCT_MD_FLAG_ALLOC, "allocation");

    for (unsigned i = 0; i < 300; ++i) {
        size = orig_size = rand() % 65536;
        if (size == 0) {
            continue;
        }

        status = uct_md_mem_alloc(pd(), &size, &address, "test", &memh);
        if (size == 0) {
            EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
            continue;
        }

        ASSERT_UCS_OK(status);
        EXPECT_GE(size, orig_size);
        EXPECT_TRUE(address != NULL);
        EXPECT_TRUE(memh != UCT_INVALID_MEM_HANDLE);

        memset(address, 0xBB, size);
        uct_md_mem_free(pd(), memh);
    }
}

UCS_TEST_P(test_pd, reg) {
    size_t size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    check_caps(UCT_MD_FLAG_REG, "registration");

    for (unsigned i = 0; i < 300; ++i) {
        size = rand() % 65536;
        if (size == 0) {
            continue;
        }

        address = malloc(size);
        ASSERT_TRUE(address != NULL);

        memset(address, 0xBB, size);

        status = uct_md_mem_reg(pd(), address, size, &memh);

        ASSERT_UCS_OK(status);
        ASSERT_TRUE(memh != UCT_INVALID_MEM_HANDLE);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        status = uct_md_mem_dereg(pd(), memh);
        ASSERT_UCS_OK(status);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        free(address);
    }
}

UCS_TEST_P(test_pd, reg_perf) {
    static const unsigned count = 10000;
    ucs_status_t status;

    check_caps(UCT_MD_FLAG_REG, "registration");

    for (size_t size = 4096; size <= 4 * 1024 * 1024; size *= 2) {
        void *ptr = malloc(size);
        ASSERT_TRUE(ptr != NULL);
        memset(ptr, 0xBB, size);

        ucs_time_t start_time = ucs_get_time();
        for (unsigned i = 0; i < count; ++i) {
            uct_mem_h memh;
            status = uct_md_mem_reg(pd(), ptr, size, &memh);
            ASSERT_UCS_OK(status);
            ASSERT_TRUE(memh != UCT_INVALID_MEM_HANDLE);

            status = uct_md_mem_dereg(pd(), memh);
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
                   posix, \
                   sysv, \
                   xpmem, \
                   cuda, \
                   ib, \
                   ugni \
                   )
#define _UCT_PD_INSTANTIATE_TEST_CASE(_test_case, _mdc_name) \
    INSTANTIATE_TEST_CASE_P(_mdc_name, _test_case, \
                            testing::ValuesIn(_test_case::enum_mds(#_mdc_name)));

UCT_PD_INSTANTIATE_TEST_CASE(test_pd)
