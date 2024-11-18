/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
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

    static std::vector<test_md_param> enum_mds(const std::string& cmpt_name);

    test_md();

    bool check_invalidate_support(unsigned reg_flags) const;

    bool is_bf_arm() const;

protected:
    virtual void init();
    virtual void cleanup();
    virtual void modify_config(const std::string& name, const std::string& value,
                               modify_config_mode_t mode);
    bool check_caps(uint64_t flags) const;
    bool check_reg_mem_type(ucs_memory_type_t mem_type);
    void alloc_memory(void **address, size_t size, char *fill,
                      ucs_memory_type_t mem_type);
    void check_memory(void *address, void *expect, size_t size,
                      ucs_memory_type_t mem_type);
    void free_memory(void *address, ucs_memory_type_t mem_type);
    static bool is_device_detected(ucs_memory_type_t mem_type);
    static void* alloc_thread(void *arg);
    ucs_status_t reg_mem(unsigned flags, void *address, size_t length,
                         uct_mem_h *memh_p);
    void test_reg_mem(unsigned access_mask, unsigned invalidate_flag);

    void test_reg_advise(size_t size, size_t advise_size,
                         size_t advice_offset, bool check_non_blocking = false);
    void test_alloc_advise(ucs_memory_type_t mem_type);

    uct_md_h md() const {
        return m_md;
    }

    const uct_md_attr_v2_t& md_attr() const {
        return m_md_attr;
    }

    typedef struct {
        test_md          *self;
        uct_completion_t comp;
    } test_md_comp_t;

    test_md_comp_t &comp() {
        return m_comp;
    }

    static void dereg_cb(uct_completion_t *comp);

    const unsigned md_flags_remote_rma = UCT_MD_MEM_ACCESS_REMOTE_PUT |
                                         UCT_MD_MEM_ACCESS_REMOTE_GET;

    size_t                        m_comp_count;

    ucs::handle<uct_md_config_t*> m_md_config;
    ucs::handle<uct_md_h>         m_md;
    uct_md_attr_v2_t              m_md_attr;
    test_md_comp_t                m_comp;
};


class test_md_non_blocking : public test_md {
protected:
    void init() override
    {
        /* ODPv1 IB feature can work only for certain DEVX configuration */
        modify_config("IB_MLX5_DEVX_OBJECTS", "dct,dcsrq", IGNORE_IF_NOT_EXIST);
        test_md::init();
    }

    void test_nb_reg_advise()
    {
        for (auto size : {UCS_KBYTE, UCS_MBYTE}) {
            test_reg_advise(size, size, 0, true);
            test_reg_advise(size, size / 2, 0, true);
            test_reg_advise(size, size / 2, size / 4, true);
        }
    }

    void test_nb_reg()
    {
        for (auto size : {UCS_KBYTE, UCS_MBYTE}) {
            test_reg_advise(size, 0, 0, true);
        }
    }
};


#define _UCT_MD_INSTANTIATE_TEST_CASE(_test_case, _cmpt_name) \
    INSTANTIATE_TEST_SUITE_P(_cmpt_name, _test_case, \
                            testing::ValuesIn(_test_case::enum_mds(#_cmpt_name)));
#endif
