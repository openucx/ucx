/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "test_md.h"

#include <common/mem_buffer.h>

#include <uct/api/uct.h>
extern "C" {
#include <ucs/time/time.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/atomic.h>
#include <ucs/sys/math.h>
}
#include <linux/sockios.h>
#include <net/if_arp.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <net/if.h>


void* test_md::alloc_thread(void *arg)
{
    volatile int *stop_flag = (int*)arg;

    while (!*stop_flag) {
        int count = ucs::rand() % 100;
        std::vector<void*> buffers;
        for (int i = 0; i < count; ++i) {
            buffers.push_back(malloc(ucs::rand() % (256 * UCS_KBYTE)));
        }
        std::for_each(buffers.begin(), buffers.end(), free);
    }
    return NULL;
}

std::vector<test_md_param> test_md::enum_mds(const std::string& cmpt_name) {

    std::vector<md_resource> md_resources = enum_md_resources();

    std::vector<test_md_param> result;
    for (std::vector<md_resource>::iterator iter = md_resources.begin();
         iter != md_resources.end(); ++iter) {
        if (iter->cmpt_attr.name == cmpt_name) {
            result.push_back(test_md_param());
            result.back().component = iter->cmpt;
            result.back().md_name   = iter->rsc_desc.md_name;
        }
    }

    return result;
}

test_md::test_md()
{
    UCS_TEST_CREATE_HANDLE(uct_md_config_t*, m_md_config,
                           (void (*)(uct_md_config_t*))uct_config_release,
                           uct_md_config_read, GetParam().component, NULL, NULL);
    memset(&m_md_attr, 0, sizeof(m_md_attr));
}

void test_md::init()
{
    ucs::test_base::init();
    UCS_TEST_CREATE_HANDLE(uct_md_h, m_md, uct_md_close, uct_md_open,
                           GetParam().component, GetParam().md_name.c_str(),
                           m_md_config);

    ucs_status_t status = uct_md_query(m_md, &m_md_attr);
    ASSERT_UCS_OK(status);

    check_skip_test();
}

void test_md::cleanup()
{
    m_md.reset();
    ucs::test_base::cleanup();
}

void test_md::modify_config(const std::string& name, const std::string& value,
                            bool optional)
{
    ucs_status_t status = uct_config_modify(m_md_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        return ucs::test_base::modify_config(name, value, optional);
    } else {
        ASSERT_UCS_OK(status);
    }
}

bool test_md::check_caps(uint64_t flags)
{
    return ((md() == NULL) || ucs_test_all_flags(m_md_attr.cap.flags, flags));
}

void test_md::alloc_memory(void **address, size_t size, char *fill_buffer,
                           ucs_memory_type_t mem_type)
{
    *address = mem_buffer::allocate(size, mem_type);
    if (fill_buffer) {
        mem_buffer::copy_to(*address, fill_buffer, size, mem_type);
    }
}

void test_md::check_memory(void *address, void *expect, size_t size,
                           ucs_memory_type_t mem_type)
{
    EXPECT_TRUE(mem_buffer::compare(expect, address, size, mem_type));
}

void test_md::free_memory(void *address, ucs_memory_type_t mem_type)
{
    mem_buffer::release(address, mem_type);
}

UCS_TEST_SKIP_COND_P(test_md, rkey_ptr,
                     !check_caps(UCT_MD_FLAG_ALLOC |
                                 UCT_MD_FLAG_RKEY_PTR)) {
    size_t size;
    uct_md_attr_t md_attr;
    void *rkey_buffer;
    ucs_status_t status;
    unsigned *rva, *lva;
    uct_mem_h memh;
    uct_rkey_bundle_t rkey_bundle;
    unsigned i;

    // alloc (should work with both sysv and xpmem
    size = sizeof(unsigned) * UCS_MBYTE;
    rva  = NULL;
    status = uct_md_mem_alloc(md(), &size, (void **)&rva,
                              UCT_MD_MEM_ACCESS_ALL,
                              "test", &memh);
    ASSERT_UCS_OK(status);
    EXPECT_LE(sizeof(unsigned) * UCS_MBYTE, size);

    // pack
    status = uct_md_query(md(), &md_attr);
    ASSERT_UCS_OK(status);
    rkey_buffer = malloc(md_attr.rkey_packed_size);
    if (rkey_buffer == NULL) {
        // make coverity happy
        uct_md_mem_free(md(), memh);
        GTEST_FAIL();
    }

    status = uct_md_mkey_pack(md(), memh, rkey_buffer);

    // unpack
    status = uct_rkey_unpack(GetParam().component, rkey_buffer, &rkey_bundle);
    ASSERT_UCS_OK(status);

    // get direct ptr
    status = uct_rkey_ptr(GetParam().component, &rkey_bundle, (uintptr_t)rva,
                          (void **)&lva);
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
    status = uct_rkey_ptr(GetParam().component, &rkey_bundle, (uintptr_t)(rva-1),
                          (void **)&lva);
    UCS_TEST_MESSAGE << "rkey_ptr of invalid address returned "
                     << ucs_status_string(status);

    status = uct_rkey_ptr(GetParam().component, &rkey_bundle, (uintptr_t)rva+size,
                          (void **)&lva);
    UCS_TEST_MESSAGE << "rkey_ptr of invalid address returned "
                     << ucs_status_string(status);

    free(rkey_buffer);
    uct_md_mem_free(md(), memh);
    uct_rkey_release(GetParam().component, &rkey_bundle);
}

UCS_TEST_SKIP_COND_P(test_md, alloc,
                     !check_caps(UCT_MD_FLAG_ALLOC)) {
    size_t size, orig_size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    for (unsigned i = 0; i < 300; ++i) {
        size = orig_size = ucs::rand() % 65536;
        if (size == 0) {
            continue;
        }

        address = NULL;
        status = uct_md_mem_alloc(md(), &size, &address,
                                  UCT_MD_MEM_ACCESS_ALL, "test", &memh);
        EXPECT_GT(size, 0ul);

        ASSERT_UCS_OK(status);
        EXPECT_GE(size, orig_size);
        EXPECT_TRUE(address != NULL);
        EXPECT_TRUE(memh != UCT_MEM_HANDLE_NULL);

        memset(address, 0xBB, size);
        uct_md_mem_free(md(), memh);
    }
}

UCS_TEST_P(test_md, mem_type_detect_mds) {

    uct_md_attr_t md_attr;
    ucs_status_t status;
    ucs_memory_type_t mem_type;
    int mem_type_id;
    void *address;

    status = uct_md_query(md(), &md_attr);
    ASSERT_UCS_OK(status);

    if (!md_attr.cap.detect_mem_types) {
        UCS_TEST_SKIP_R("MD can't detect any memory types");
    }

    ucs_for_each_bit(mem_type_id, md_attr.cap.detect_mem_types) {
        alloc_memory(&address, UCS_KBYTE, NULL,
                     static_cast<ucs_memory_type_t>(mem_type_id));
        status = uct_md_detect_memory_type(md(), address, 1024, &mem_type);
        ASSERT_UCS_OK(status);
        EXPECT_TRUE(mem_type == mem_type_id);
    }
}

UCS_TEST_SKIP_COND_P(test_md, reg,
                     !check_caps(UCT_MD_FLAG_REG)) {
    size_t size;
    uct_md_attr_t md_attr;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    status = uct_md_query(md(), &md_attr);
    ASSERT_UCS_OK(status);
    for (unsigned mem_type_id = 0; mem_type_id < UCS_MEMORY_TYPE_LAST; mem_type_id++) {
        ucs_memory_type_t mem_type = static_cast<ucs_memory_type_t>(mem_type_id);

        if (!(md_attr.cap.reg_mem_types & UCS_BIT(mem_type_id))) {
            UCS_TEST_MESSAGE << mem_buffer::mem_type_name(mem_type) << " memory "
                             << "registration is not supported by "
                             << GetParam().md_name;
            continue;
        }

        for (unsigned i = 0; i < 300; ++i) {
            size = ucs::rand() % 65536;
            if (size == 0) {
                continue;
            }

            std::vector<char> fill_buffer(size, 0);
            ucs::fill_random(fill_buffer);

            alloc_memory(&address, size, &fill_buffer[0], mem_type);

            status = uct_md_mem_reg(md(), address, size, UCT_MD_MEM_ACCESS_ALL, &memh);

            ASSERT_UCS_OK(status);
            ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);
            check_memory(address, &fill_buffer[0], size, mem_type);

            status = uct_md_mem_dereg(md(), memh);
            ASSERT_UCS_OK(status);
            check_memory(address, &fill_buffer[0], size, mem_type);

            free_memory(address, mem_type);

        }
    }
}

UCS_TEST_SKIP_COND_P(test_md, reg_perf,
                     !check_caps(UCT_MD_FLAG_REG)) {
    static const unsigned count = 10000;
    ucs_status_t status;
    uct_md_attr_t md_attr;
    void *ptr;

    status = uct_md_query(md(), &md_attr);
    ASSERT_UCS_OK(status);
    for (unsigned mem_type_id = 0; mem_type_id < UCS_MEMORY_TYPE_LAST; mem_type_id++) {
        ucs_memory_type_t mem_type = static_cast<ucs_memory_type_t>(mem_type_id);
        if (!(md_attr.cap.reg_mem_types & UCS_BIT(mem_type_id))) {
            UCS_TEST_MESSAGE << mem_buffer::mem_type_name(mem_type) << " memory "
                             << " registration is not supported by "
                             << GetParam().md_name;
            continue;
        }
        for (size_t size = 4 * UCS_KBYTE; size <= 4 * UCS_MBYTE; size *= 2) {
            alloc_memory(&ptr, size, NULL,
                         static_cast<ucs_memory_type_t>(mem_type_id));

            ucs_time_t start_time = ucs_get_time();
            ucs_time_t end_time = start_time;

            unsigned n = 0;
            while (n < count) {
                uct_mem_h memh;
                status = uct_md_mem_reg(md(), ptr, size, UCT_MD_MEM_ACCESS_ALL,
                        &memh);
                ASSERT_UCS_OK(status);
                ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

                status = uct_md_mem_dereg(md(), memh);
                ASSERT_UCS_OK(status);

                ++n;
                end_time = ucs_get_time();

                if (end_time - start_time > ucs_time_from_sec(1.0)) {
                    break;
                }
            }

            UCS_TEST_MESSAGE << GetParam().md_name << ": Registration time for " <<
                ucs_memory_type_names[mem_type] << " memory " << size << " bytes: " <<
                long(ucs_time_to_nsec(end_time - start_time) / n) << " ns";

            free_memory(ptr, mem_type);
        }
    }
}

UCS_TEST_SKIP_COND_P(test_md, reg_advise,
                     !check_caps(UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_ADVISE)) {
    size_t size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    size = 128 * UCS_MBYTE;
    address = malloc(size);
    ASSERT_TRUE(address != NULL);

    status = uct_md_mem_reg(md(), address, size,
                            UCT_MD_MEM_FLAG_NONBLOCK|UCT_MD_MEM_ACCESS_ALL,
                            &memh);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    status = uct_md_mem_advise(md(), memh, (char *)address + 7,
                               32 * UCS_KBYTE, UCT_MADV_WILLNEED);
    EXPECT_UCS_OK(status);

    status = uct_md_mem_dereg(md(), memh);
    EXPECT_UCS_OK(status);
    free(address);
}

UCS_TEST_SKIP_COND_P(test_md, alloc_advise,
                     !check_caps(UCT_MD_FLAG_ALLOC |
                                 UCT_MD_FLAG_ADVISE)) {
    size_t size, orig_size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    orig_size = size = 128 * UCS_MBYTE;
    address   = NULL;

    status = uct_md_mem_alloc(md(), &size, &address,
                              UCT_MD_MEM_FLAG_NONBLOCK|
                              UCT_MD_MEM_ACCESS_ALL,
                              "test", &memh);
    ASSERT_UCS_OK(status);
    EXPECT_GE(size, orig_size);
    EXPECT_TRUE(address != NULL);
    EXPECT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    status = uct_md_mem_advise(md(), memh, (char *)address + 7,
                               32 * UCS_KBYTE, UCT_MADV_WILLNEED);
    EXPECT_UCS_OK(status);

    memset(address, 0xBB, size);
    uct_md_mem_free(md(), memh);
}

/*
 * reproduce issue #1284, main thread is registering memory while another thread
 * allocates and releases memory.
 */
UCS_TEST_SKIP_COND_P(test_md, reg_multi_thread,
                     !check_caps(UCT_MD_FLAG_REG)) {
    ucs_status_t status;
    uct_md_attr_t md_attr;

    status = uct_md_query(md(), &md_attr);
    ASSERT_UCS_OK(status);

    if (!(md_attr.cap.reg_mem_types & UCS_BIT(UCS_MEMORY_TYPE_HOST))) {
        UCS_TEST_SKIP_R("not host memory type");
    }

    pthread_t thread_id;
    int stop_flag = 0;
    pthread_create(&thread_id, NULL, alloc_thread, &stop_flag);

    ucs_time_t start_time = ucs_get_time();
    while (ucs_get_time() - start_time < ucs_time_from_sec(0.5)) {
        const size_t size = (ucs::rand() % 65536) + 1;

        void *buffer = malloc(size);
        ASSERT_TRUE(buffer != NULL);

        uct_mem_h memh;
        status = uct_md_mem_reg(md(), buffer, size,
                                UCT_MD_MEM_FLAG_NONBLOCK|
                                UCT_MD_MEM_ACCESS_ALL,
                                &memh);
        ASSERT_UCS_OK(status, << " buffer=" << buffer << " size=" << size);
        ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

        sched_yield();

        status = uct_md_mem_dereg(md(), memh);
        EXPECT_UCS_OK(status);
        free(buffer);
    }

    stop_flag = 1;
    pthread_join(thread_id, NULL);
}

UCS_TEST_SKIP_COND_P(test_md, sockaddr_accessibility,
                     !check_caps(UCT_MD_FLAG_SOCKADDR)) {
    ucs_sock_addr_t sock_addr;
    struct ifaddrs *ifaddr, *ifa;
    bool found_rdma = false;
    bool found_ip   = false;

    ASSERT_TRUE(getifaddrs(&ifaddr) != -1);

    /* go through a linked list of available interfaces */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ucs::is_inet_addr(ifa->ifa_addr) &&
            ucs_netif_flags_is_active(ifa->ifa_flags)) {
            sock_addr.addr = ifa->ifa_addr;

            found_ip = true;

            if (GetParam().md_name == "rdmacm") {
                if (ucs::is_rdmacm_netdev(ifa->ifa_name)) {
                    UCS_TEST_MESSAGE << "Testing " << ifa->ifa_name << " with " <<
                                        ucs::sockaddr_to_str(ifa->ifa_addr);
                    ASSERT_TRUE(uct_md_is_sockaddr_accessible(md(), &sock_addr,
                                                              UCT_SOCKADDR_ACC_LOCAL));
                    ASSERT_TRUE(uct_md_is_sockaddr_accessible(md(), &sock_addr,
                                                              UCT_SOCKADDR_ACC_REMOTE));
                    found_rdma = true;
                }
            } else {
                UCS_TEST_MESSAGE << "Testing " << ifa->ifa_name << " with " <<
                                    ucs::sockaddr_to_str(ifa->ifa_addr);
                ASSERT_TRUE(uct_md_is_sockaddr_accessible(md(), &sock_addr,
                                                          UCT_SOCKADDR_ACC_LOCAL));
                ASSERT_TRUE(uct_md_is_sockaddr_accessible(md(), &sock_addr,
                                                          UCT_SOCKADDR_ACC_REMOTE));
            }
        }
    }

    if (GetParam().md_name == "rdmacm") {
        if (!found_rdma) {
            UCS_TEST_MESSAGE <<
                "Cannot find an IPoIB/RoCE interface with an IPv4 address on the host";
        }
    } else if (!found_ip) {
        UCS_TEST_MESSAGE << "Cannot find an IPv4/IPv6 interface on the host";
    }

    freeifaddrs(ifaddr);
}

#define UCT_MD_INSTANTIATE_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_MD_INSTANTIATE_TEST_CASE, _test_case, \
                   knem, \
                   cma, \
                   posix, \
                   sysv, \
                   xpmem, \
                   cuda_cpy, \
                   cuda_ipc, \
                   rocm_cpy, \
                   rocm_ipc, \
                   ib, \
                   ugni, \
                   sockcm, \
                   rdmacm \
                   )

UCT_MD_INSTANTIATE_TEST_CASE(test_md)

