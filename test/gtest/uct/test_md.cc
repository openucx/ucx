/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <common/test.h>

class test_md : public testing::TestWithParam<std::string>,
                public ucs::test_base
{
public:
    UCS_TEST_BASE_IMPL;

    static std::vector<std::string> enum_mds(const std::string& mdc_name);

    test_md();

protected:
    virtual void init();
    virtual void cleanup();
    virtual void modify_config(const std::string& name, const std::string& value);
    void check_caps(uint64_t flags, const std::string& name);

    void test_registration();

    uct_md_h pd() {
        return m_pd;
    }

    static void* alloc_thread(void *arg)
    {
        volatile int *stop_flag = (int*)arg;

        while (!*stop_flag) {
            int count = ucs::rand() % 100;
            std::vector<void*> buffers;
            for (int i = 0; i < count; ++i) {
                buffers.push_back(malloc(ucs::rand() % (256*1024)));
            }
            std::for_each(buffers.begin(), buffers.end(), free);
        }
        return NULL;
    }

private:
    ucs::handle<uct_md_config_t*> m_md_config;
    ucs::handle<uct_md_h>         m_pd;
};

std::vector<std::string> test_md::enum_mds(const std::string& mdc_name) {
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

test_md::test_md()
{
    UCS_TEST_CREATE_HANDLE(uct_md_config_t*, m_md_config,
                           (void (*)(uct_md_config_t*))uct_config_release,
                           uct_md_config_read, GetParam().c_str(), NULL, NULL);
}

void test_md::init()
{
    ucs::test_base::init();
    UCS_TEST_CREATE_HANDLE(uct_md_h, m_pd, uct_md_close, uct_md_open,
                           GetParam().c_str(), m_md_config);
}

void test_md::cleanup()
{
    m_pd.reset();
    ucs::test_base::cleanup();
}

void test_md::modify_config(const std::string& name, const std::string& value)
{
    ucs_status_t status = uct_config_modify(m_md_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        return ucs::test_base::modify_config(name, value);
    } else {
        ASSERT_UCS_OK(status);
    }
}

void test_md::check_caps(uint64_t flags, const std::string& name)
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

UCS_TEST_P(test_md, rkey_ptr) {

    size_t size;
    uct_md_attr_t md_attr;
    void *rkey_buffer;
    ucs_status_t status;
    unsigned *rva, *lva;
    uct_mem_h memh;
    uct_rkey_bundle_t rkey_bundle;
    unsigned i;

    check_caps(UCT_MD_FLAG_ALLOC|UCT_MD_FLAG_DIRECT_ACCESS, "allocation+direct access");
    // alloc (should work with both sysv and xpmem
    size = 1024 * 1024 * sizeof(unsigned);
    status = uct_md_mem_alloc(pd(), &size, (void **)&rva, 0, "test", &memh);
    ASSERT_UCS_OK(status);
    EXPECT_LE(1024 * 1024 * sizeof(unsigned), size);

    // pack
    status = uct_md_query(pd(), &md_attr);
    ASSERT_UCS_OK(status);
    rkey_buffer = malloc(md_attr.rkey_packed_size);
    ASSERT_TRUE(rkey_buffer != NULL);
    status = uct_md_mkey_pack(pd(), memh, rkey_buffer);

    // unpack
    status = uct_rkey_unpack(rkey_buffer, &rkey_bundle);
    ASSERT_UCS_OK(status);

    // get direct ptr
    status = uct_rkey_ptr(&rkey_bundle, rva, (void **)&lva);
    ASSERT_UCS_OK(status);
    // check direct access
    // read
    for (i = 0; i < size/sizeof(unsigned); i++) {
        rva[i] = i;
    }
    EXPECT_EQ(memcmp(lva, rva, size), 0);

    // write
    for (i = 0; i < size/sizeof(unsigned); i++) {
        lva[i] = size - i;
    }
    EXPECT_EQ(memcmp(lva, rva, size), 0);

    // check bounds
    //
    status = uct_rkey_ptr(&rkey_bundle, rva-1, (void **)&lva);
    EXPECT_EQ(UCS_ERR_INVALID_ADDR, status);

    status = uct_rkey_ptr(&rkey_bundle, (char *)rva+size, (void **)&lva);
    EXPECT_EQ(UCS_ERR_INVALID_ADDR, status);

    free(rkey_buffer);
    uct_md_mem_free(pd(), memh);
    uct_rkey_release(&rkey_bundle);
}

UCS_TEST_P(test_md, alloc) {
    size_t size, orig_size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    check_caps(UCT_MD_FLAG_ALLOC, "allocation");

    for (unsigned i = 0; i < 300; ++i) {
        size = orig_size = ucs::rand() % 65536;
        if (size == 0) {
            continue;
        }

        status = uct_md_mem_alloc(pd(), &size, &address, 0, "test", &memh);
        EXPECT_GT(size, 0ul);

        ASSERT_UCS_OK(status);
        EXPECT_GE(size, orig_size);
        EXPECT_TRUE(address != NULL);
        EXPECT_TRUE(memh != UCT_MEM_HANDLE_NULL);

        memset(address, 0xBB, size);
        uct_md_mem_free(pd(), memh);
    }
}

UCS_TEST_P(test_md, reg) {
    size_t size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    check_caps(UCT_MD_FLAG_REG, "registration");

    for (unsigned i = 0; i < 300; ++i) {
        size = ucs::rand() % 65536;
        if (size == 0) {
            continue;
        }

        address = malloc(size);
        ASSERT_TRUE(address != NULL);

        memset(address, 0xBB, size);

        status = uct_md_mem_reg(pd(), address, size, 0, &memh);

        ASSERT_UCS_OK(status);
        ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        status = uct_md_mem_dereg(pd(), memh);
        ASSERT_UCS_OK(status);
        EXPECT_EQ('\xBB', *((char*)address + size - 1));

        free(address);
    }
}

UCS_TEST_P(test_md, reg_perf) {
    static const unsigned count = 10000;
    ucs_status_t status;

    check_caps(UCT_MD_FLAG_REG, "registration");

    for (size_t size = 4096; size <= 4 * 1024 * 1024; size *= 2) {
        void *ptr = malloc(size);
        ASSERT_TRUE(ptr != NULL);
        memset(ptr, 0xBB, size);

        ucs_time_t start_time = ucs_get_time();
        ucs_time_t end_time = start_time;

        unsigned n = 0;
        while (n < count) {
            uct_mem_h memh;
            status = uct_md_mem_reg(pd(), ptr, size, 0, &memh);
            ASSERT_UCS_OK(status);
            ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

            status = uct_md_mem_dereg(pd(), memh);
            ASSERT_UCS_OK(status);

            ++n;
            end_time = ucs_get_time();

            if (end_time - start_time > ucs_time_from_sec(1.0)) {
                break;
            }
        }

        UCS_TEST_MESSAGE << GetParam() << ": Registration time for " <<
                        size << " bytes: " <<
                        long(ucs_time_to_nsec(end_time - start_time) / n) << " ns";

        free(ptr);
    }
}

UCS_TEST_P(test_md, reg_advise) {
    size_t size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    check_caps(UCT_MD_FLAG_REG|UCT_MD_FLAG_ADVISE, "registration&advise");

    size = 128 * 1024 * 1024;
    address = malloc(size);
    ASSERT_TRUE(address != NULL);

    status = uct_md_mem_reg(pd(), address, size, UCT_MD_MEM_FLAG_NONBLOCK, &memh);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    status = uct_md_mem_advise(pd(), memh, (char *)address + 7, 32*1024, UCT_MADV_WILLNEED);
    EXPECT_UCS_OK(status);

    status = uct_md_mem_dereg(pd(), memh);
    EXPECT_UCS_OK(status);
    free(address);
}

UCS_TEST_P(test_md, alloc_advise) {
    size_t size, orig_size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    check_caps(UCT_MD_FLAG_ALLOC|UCT_MD_FLAG_ADVISE, "allocation&advise");

    orig_size = size = 128 * 1024 * 1024;

    status = uct_md_mem_alloc(pd(), &size, &address, UCT_MD_MEM_FLAG_NONBLOCK, "test", &memh);
    ASSERT_UCS_OK(status);
    EXPECT_GE(size, orig_size);
    EXPECT_TRUE(address != NULL);
    EXPECT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    status = uct_md_mem_advise(pd(), memh, (char *)address + 7, 32*1024, UCT_MADV_WILLNEED);
    EXPECT_UCS_OK(status);

    memset(address, 0xBB, size);
    uct_md_mem_free(pd(), memh);
}

/*
 * reproduce issue #1284, main thread is registering memory while another thread
 * allocates and releases memory.
 */
UCS_TEST_P(test_md, reg_multi_thread) {
    ucs_status_t status;

    check_caps(UCT_MD_FLAG_REG, "registration");

    pthread_t thread_id;
    int stop_flag = 0;
    pthread_create(&thread_id, NULL, alloc_thread, &stop_flag);

    ucs_time_t start_time = ucs_get_time();
    while (ucs_get_time() - start_time < ucs_time_from_sec(0.5)) {
        const size_t size = (ucs::rand() % 65536) + 1;

        void *buffer = malloc(size);
        ASSERT_TRUE(buffer != NULL);

        uct_mem_h memh;
        status = uct_md_mem_reg(pd(), buffer, size, UCT_MD_MEM_FLAG_NONBLOCK, &memh);
        ASSERT_UCS_OK(status, << " buffer=" << buffer << " size=" << size);
        ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

        sched_yield();

        status = uct_md_mem_dereg(pd(), memh);
        EXPECT_UCS_OK(status);
        free(buffer);
    }

    stop_flag = 1;
    pthread_join(thread_id, NULL);
}


#define UCT_PD_INSTANTIATE_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_PD_INSTANTIATE_TEST_CASE, _test_case, \
                   knem, \
                   cma, \
                   posix, \
                   sysv, \
                   xpmem, \
                   cuda, \
                   rocm, \
                   ib, \
                   ugni \
                   )
#define _UCT_PD_INSTANTIATE_TEST_CASE(_test_case, _mdc_name) \
    INSTANTIATE_TEST_CASE_P(_mdc_name, _test_case, \
                            testing::ValuesIn(_test_case::enum_mds(#_mdc_name)));

UCT_PD_INSTANTIATE_TEST_CASE(test_md)
