/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCT_TEST_MD
#define UCT_TEST_MD

#include "uct_test.h"


struct test_md_param {
    uct_component_h  component;
    std::string      md_name;
};

static std::ostream& operator<<(std::ostream& os, const test_md_param& md_param) {
    return os << md_param.md_name;
}

class test_md : public testing::TestWithParam<test_md_param>,
                public uct_test_base
{
public:
    UCS_TEST_BASE_IMPL;

    static std::vector<test_md_param> enum_mds(const std::string& mdc_name);

    test_md();

protected:
    virtual void init();
    virtual void cleanup();
    virtual void modify_config(const std::string& name, const std::string& value,
                               bool optional);
    void check_caps(uint64_t flags, const std::string& name);
    void alloc_memory(void **address, size_t size, char *fill, int mem_type);
    void check_memory(void *address, void *expect, size_t size, int mem_type);
    void free_memory(void *address, int mem_type);

    void test_registration();

    uct_md_h md() const {
        return m_md;
    }

    const uct_md_attr_t& md_attr() const {
        return m_md_attr;
    }


    static void* alloc_thread(void *arg);

private:
    ucs::handle<uct_md_config_t*> m_md_config;
    ucs::handle<uct_md_h>         m_md;
    uct_md_attr_t                 m_md_attr;
};


#define _UCT_MD_INSTANTIATE_TEST_CASE(_test_case, _mdc_name) \
    INSTANTIATE_TEST_CASE_P(_mdc_name, _test_case, \
                            testing::ValuesIn(_test_case::enum_mds(#_mdc_name)));
#endif
