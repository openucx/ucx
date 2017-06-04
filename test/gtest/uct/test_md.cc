/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/base/ib_md.h>
#include <ucs/time/time.h>
}
#include <common/test.h>

#define  IBV_EXP_ODP_SUPPORT_IMPLICIT  1 << 1

class test_md : public testing::TestWithParam<std::tr1::tuple<std::string, std::string> >,
                public ucs::test_base
{
public:
    UCS_TEST_BASE_IMPL;

    static std::vector<std::string> enum_mds(const std::string& mdc_name);

    test_md();

    std::string                   tl;

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
    int                           support_odp;
    std::string                   reg_method;
    int                           skip_test;
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
    skip_test = 0;
    support_odp = 0;
    tl = std::tr1::get<0>(GetParam());
    reg_method = std::tr1::get<1>(GetParam());
    UCS_TEST_CREATE_HANDLE(uct_md_config_t*, m_md_config,
                           (void (*)(uct_md_config_t*))uct_config_release,
                           uct_md_config_read, tl.c_str(), NULL, NULL);
}

void test_md::init()
{
    std::string tl;

    tl = std::tr1::get<0>(GetParam());
    reg_method = std::tr1::get<1>(GetParam());
    struct ibv_device **dev_list;
    struct ibv_device **orig_dev_list;
    int num_devices;

    skip_test = 0;

    support_odp = 0;
#if HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_ODP_CAPS
    if (reg_method == "odp") {
        orig_dev_list = dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list) {
            UCS_TEST_ABORT("Failed to get the device list.");
        }

        while (*dev_list) {
            if (tl ==  (std::string("ib/") + ibv_get_device_name(*dev_list)))  {
                struct ibv_exp_device_attr device_attr;
                struct ibv_device_attr device_legacy_attr;
                struct ibv_context *ctx;
                int ret;

                ctx = ibv_open_device(*dev_list);
                if (!ctx) {
                   fprintf(stderr, "Failed to open device\n");
                }

                memset(&device_attr, 0, sizeof(device_attr));
                device_attr.comp_mask = IBV_EXP_DEVICE_ATTR_RESERVED - 1;
                if (ibv_exp_query_device(ctx, &device_attr)) {
                    ret = ibv_query_device(ctx, &device_legacy_attr);
                    if (ret) {
                       ucs_error("query device failed %d: %m", ret);
                    }

                    ASSERT_TRUE(ret == 0); 
                    memcpy(&device_attr, &device_legacy_attr, sizeof(device_legacy_attr));
                }
                ibv_close_device(ctx);

                if (device_attr.odp_caps.general_odp_caps &
                    IBV_EXP_ODP_SUPPORT_IMPLICIT) {
                    support_odp = 1;
                }

                break;
            }
            ++dev_list;
        }
        ibv_free_device_list(orig_dev_list);
    }
#else
    skip_test = 1;
#endif
    ucs::test_base::init();
    push_config();

    if (tl.substr(0, 2) == "ib") {
        if (reg_method == "odp")  {
            if (support_odp) {
                modify_config("REG_METHODS" , reg_method);
            } else {
                skip_test = 1;
            }
        } else {
            modify_config("REG_METHODS" , reg_method);
        }
    } else {
        /* Avoid test duplication - test other then ib only if direct */
        if (reg_method != "direct") {
            skip_test = 1;
        }
    }

    UCS_TEST_CREATE_HANDLE(uct_md_h, m_pd, uct_md_close, uct_md_open,
                           tl.c_str(), m_md_config);
}

void test_md::cleanup()
{
    pop_config();
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
    std::stringstream ss;
    ASSERT_UCS_OK(status);
    if (!ucs_test_all_flags(md_attr.cap.flags, flags)) {
        ss << name << " is not supported by " << tl;
        UCS_TEST_SKIP_R(ss.str());
    }

    if (skip_test) {
        ss << name << " skip test " << tl;
        UCS_TEST_SKIP_R(ss.str());
    }

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

        UCS_TEST_MESSAGE << tl << ": Registration time for " <<
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
                            testing::Combine(testing::ValuesIn(_test_case::enum_mds(#_mdc_name)), \
                                             testing::Values("odp","rcache","direct")));

UCT_PD_INSTANTIATE_TEST_CASE(test_md)
