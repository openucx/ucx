/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "test_md.h"

#include <common/mem_buffer.h>

#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
extern "C" {
#include <ucs/time/time.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/atomic.h>
#include <ucs/sys/math.h>
}
#include <net/if_arp.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <net/if.h>


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

void* test_md::alloc_thread(void *arg)
{
    volatile int *stop_flag = (int*)arg;

    while (!*stop_flag) {
        int count = ucs::rand() % 100;
        ucs::ptr_vector<void> buffers;
        for (int i = 0; i < count; ++i) {
            // allocate via malloc(), because ptr_vector<void>::release()
            // method specialization uses free() to release a memory obtained
            // for an element
            buffers.push_back(malloc(ucs::rand() % (256 * UCS_KBYTE)));
        }
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
    /* coverity[uninit_member] */
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
                            modify_config_mode_t mode)
{
    ucs_status_t status = uct_config_modify(m_md_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        return ucs::test_base::modify_config(name, value, mode);
    } else {
        ASSERT_UCS_OK(status);
    }
}

bool test_md::check_caps(uint64_t flags)
{
    return ((md() == NULL) || ucs_test_all_flags(m_md_attr.cap.flags, flags));
}

bool test_md::check_reg_mem_type(ucs_memory_type_t mem_type)
{
    return ((md() == NULL) || (check_caps(UCT_MD_FLAG_REG) &&
                (m_md_attr.cap.reg_mem_types & UCS_BIT(mem_type))));
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

bool test_md::is_device_detected(ucs_memory_type_t mem_type)
{
    return (mem_type != UCS_MEMORY_TYPE_ROCM) &&
           (mem_type != UCS_MEMORY_TYPE_ROCM_MANAGED);
}

void test_md::dereg_cb(uct_completion_t *comp)
{
    test_md_comp_t *md_comp = ucs_container_of(comp, test_md_comp_t, comp);

    md_comp->self->m_comp_count++;
}

UCS_TEST_SKIP_COND_P(test_md, rkey_ptr,
                     !check_caps(UCT_MD_FLAG_ALLOC |
                                 UCT_MD_FLAG_RKEY_PTR)) {
    uct_md_h md_ref           = md();
    uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
    size_t size;
    void *rkey_buffer;
    ucs_status_t status;
    unsigned *rva, *lva;
    uct_allocated_memory_t mem;
    uct_rkey_bundle_t rkey_bundle;
    unsigned i;
    uct_mem_alloc_params_t params;

    params.field_mask      = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS    |
                             UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS  |
                             UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                             UCT_MEM_ALLOC_PARAM_FIELD_MDS      |
                             UCT_MEM_ALLOC_PARAM_FIELD_NAME;
    params.flags           = UCT_MD_MEM_ACCESS_ALL;
    params.name            = "test";
    params.mem_type        = UCS_MEMORY_TYPE_HOST;
    params.mds.mds         = &md_ref;
    params.mds.count       = 1;

    // alloc (should work with both sysv and xpmem
    size             = sizeof(unsigned) * UCS_MBYTE;
    rva              = NULL;
    params.address   = (void *)rva;
    status = uct_mem_alloc(size, &method, 1, &params, &mem);
    ASSERT_UCS_OK(status);
    EXPECT_LE(sizeof(unsigned) * UCS_MBYTE, mem.length);
    size   = mem.length;
    rva    = (unsigned *)mem.address;

    // pack
    rkey_buffer = malloc(md_attr().rkey_packed_size);
    if (rkey_buffer == NULL) {
        // make coverity happy
        uct_mem_free(&mem);
        GTEST_FAIL();
    }

    status = uct_md_mkey_pack(md(), mem.memh, rkey_buffer);

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
    uct_mem_free(&mem);
    uct_rkey_release(GetParam().component, &rkey_bundle);
}

UCS_TEST_SKIP_COND_P(test_md, alloc,
                     !check_caps(UCT_MD_FLAG_ALLOC)) {
    uct_md_h md_ref           = md();
    uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
    size_t size, orig_size;
    ucs_status_t status;
    void *address;
    unsigned mem_type;
    uct_allocated_memory_t mem;
    uct_mem_alloc_params_t params;

    params.field_mask      = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS    |
                             UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS  |
                             UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                             UCT_MEM_ALLOC_PARAM_FIELD_MDS      |
                             UCT_MEM_ALLOC_PARAM_FIELD_NAME;
    params.flags           = UCT_MD_MEM_ACCESS_ALL;
    params.name            = "test";
    params.mds.mds         = &md_ref;
    params.mds.count       = 1;

    ucs_for_each_bit(mem_type, md_attr().cap.alloc_mem_types) {
        for (unsigned i = 0; i < 300; ++i) {
            size = orig_size = ucs::rand() % 65536;
            if (size == 0) {
                continue;
            }

            address         = NULL;
            params.address  = address;
            params.mem_type = (ucs_memory_type_t)mem_type;

            status = uct_mem_alloc(size, &method, 1, &params, &mem);

            EXPECT_GT(mem.length, 0ul);
            address = mem.address;
            size    = mem.length;

            ASSERT_UCS_OK(status);
            EXPECT_GE(size, orig_size);
            EXPECT_TRUE(address != NULL);
            EXPECT_TRUE(mem.memh != UCT_MEM_HANDLE_NULL);

            if (mem_type == UCS_MEMORY_TYPE_HOST) {
                memset(address, 0xBB, size);
            }
            uct_mem_free(&mem);
        }
    }
}

UCS_TEST_P(test_md, mem_type_detect_mds) {
    const size_t buffer_size = 1024;
    size_t slice_offset;
    size_t slice_length;
    ucs_status_t status;
    int alloc_mem_type;
    void *address;

    if (!md_attr().cap.detect_mem_types) {
        UCS_TEST_SKIP_R("MD can't detect any memory types");
    }

    ucs_for_each_bit(alloc_mem_type, md_attr().cap.detect_mem_types) {
        ucs_assert(alloc_mem_type < UCS_MEMORY_TYPE_LAST); /* for coverity */

        alloc_memory(&address, buffer_size, NULL,
                     static_cast<ucs_memory_type_t>(alloc_mem_type));

        /* test legacy detect_memory_type API */
        ucs_memory_type_t detected_mem_type;
        status = uct_md_detect_memory_type(md(), address, buffer_size,
                                           &detected_mem_type);
        ASSERT_UCS_OK(status);
        EXPECT_EQ(alloc_mem_type, detected_mem_type);

        /* test mem_query API */
        uct_md_mem_attr_t mem_attr;
        mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE     |
                              UCT_MD_MEM_ATTR_FIELD_SYS_DEV      |
                              UCT_MD_MEM_ATTR_FIELD_BASE_ADDRESS |
                              UCT_MD_MEM_ATTR_FIELD_ALLOC_LENGTH;

        for (unsigned i = 0; i < 300; i++) {
            slice_offset = ucs::rand() % buffer_size;
            slice_length = ucs::rand() % buffer_size;

            if (slice_length == 0) {
                continue;
            }

            status = uct_md_mem_query(md(),
                                      UCS_PTR_BYTE_OFFSET(address,
                                                          slice_offset),
                                      slice_length, &mem_attr);
            ASSERT_UCS_OK(status);
            EXPECT_EQ(alloc_mem_type, mem_attr.mem_type);
            if (alloc_mem_type == UCS_MEMORY_TYPE_CUDA) {
                EXPECT_EQ(buffer_size, mem_attr.alloc_length);
                EXPECT_EQ(address, mem_attr.base_address);
            } else {
                EXPECT_EQ(slice_length, mem_attr.alloc_length);
                EXPECT_EQ(UCS_PTR_BYTE_OFFSET(address, slice_offset),
                          mem_attr.base_address);
            }
        }

        /* print memory type and dev name */
        mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_SYS_DEV;
        status = uct_md_mem_query(md(), address, buffer_size, &mem_attr);
        ASSERT_UCS_OK(status);

        const char *dev_name = ucs_topo_sys_device_get_name(mem_attr.sys_dev);
        UCS_TEST_MESSAGE << ucs_memory_type_names[alloc_mem_type] << ": "
                         << "sys_dev[" << static_cast<int>(mem_attr.sys_dev)
                         << "] (" << dev_name << ")";

        free_memory(address, static_cast<ucs_memory_type_t>(alloc_mem_type));
    }
}

UCS_TEST_P(test_md, mem_query) {
    for (auto mem_type : mem_buffer::supported_mem_types()) {
        if (!(md_attr().cap.detect_mem_types & UCS_BIT(mem_type))) {
            continue;
        }

        mem_buffer mem_buf(4 * UCS_KBYTE, mem_type);
        uct_md_mem_attr_t mem_attr = {};

        mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE |
                              UCT_MD_MEM_ATTR_FIELD_SYS_DEV;
        ucs_status_t status = uct_md_mem_query(md(), mem_buf.ptr(),
                                               mem_buf.size(), &mem_attr);
        ASSERT_UCS_OK(status);
        EXPECT_EQ(mem_type, mem_attr.mem_type);
        if (is_device_detected(mem_attr.mem_type)) {
            EXPECT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, mem_attr.sys_dev);
        }

        UCS_TEST_MESSAGE << ucs_memory_type_names[mem_type] << ": "
                         << ucs_topo_sys_device_get_name(mem_attr.sys_dev);
    }
}

UCS_TEST_P(test_md, sys_device) {
    uct_tl_resource_desc_t *tl_resources;
    unsigned num_tl_resources;

    ucs_status_t status = uct_md_query_tl_resources(md(), &tl_resources,
                                                    &num_tl_resources);
    ASSERT_UCS_OK(status);

    for (unsigned i = 0; i < num_tl_resources; ++i) {
        const char *sysdev_name = ucs_topo_sys_device_get_name(
                tl_resources[i].sys_device);
        ASSERT_TRUE(sysdev_name != NULL);
        UCS_TEST_MESSAGE << tl_resources[i].dev_name << ": " << sysdev_name;

        /* Expect 0 latency and infinite bandwidth within same device */
        ucs_sys_dev_distance_t distance;
        ucs_topo_get_distance(tl_resources[i].sys_device,
                              tl_resources[i].sys_device,
                              &distance);
        EXPECT_NEAR(distance.latency, 0, 1e-9);
        EXPECT_GT(distance.bandwidth, 1e12);

        /* Expect real device detection on IB transports */
        if (!strcmp(md_attr().component_name, "ib")) {
            EXPECT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, tl_resources[i].sys_device);
        }
    }

    uct_release_tl_resource_list(tl_resources);
}

UCS_TEST_SKIP_COND_P(test_md, reg,
                     !check_caps(UCT_MD_FLAG_REG)) {
    size_t size;
    ucs_status_t status;
    void *address;
    uct_mem_h memh;

    for (unsigned mem_type_id = 0; mem_type_id < UCS_MEMORY_TYPE_LAST; mem_type_id++) {
        ucs_memory_type_t mem_type = static_cast<ucs_memory_type_t>(mem_type_id);
        if (!(md_attr().cap.reg_mem_types & UCS_BIT(mem_type_id))) {
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
    void *ptr;

    for (unsigned mem_type_id = 0; mem_type_id < UCS_MEMORY_TYPE_LAST; mem_type_id++) {
        ucs_memory_type_t mem_type = static_cast<ucs_memory_type_t>(mem_type_id);
        if (!(md_attr().cap.reg_mem_types & UCS_BIT(mem_type_id))) {
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
    uct_md_h md_ref           = md();
    uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
    void *address             = NULL;
    size_t size, orig_size;
    ucs_status_t status;
    uct_allocated_memory_t mem;
    uct_mem_alloc_params_t params;

    params.field_mask      = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS    |
                             UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS  |
                             UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                             UCT_MEM_ALLOC_PARAM_FIELD_MDS      |
                             UCT_MEM_ALLOC_PARAM_FIELD_NAME;
    params.flags           = UCT_MD_MEM_FLAG_NONBLOCK | UCT_MD_MEM_ACCESS_ALL;
    params.name            = "test";
    params.mem_type        = UCS_MEMORY_TYPE_HOST;
    params.address         = address;
    params.mds.mds         = &md_ref;
    params.mds.count       = 1;

    size          = 128 * UCS_MBYTE;
    orig_size     = size;

    status  = uct_mem_alloc(size, &method, 1, &params, &mem);
    address = mem.address;
    size    = mem.length;
    ASSERT_UCS_OK(status);
    EXPECT_GE(size, orig_size);
    EXPECT_TRUE(address != NULL);
    EXPECT_TRUE(mem.memh != UCT_MEM_HANDLE_NULL);

    status = uct_md_mem_advise(md(), mem.memh, (char *)address + 7,
                               32 * UCS_KBYTE, UCT_MADV_WILLNEED);
    EXPECT_UCS_OK(status);

    memset(address, 0xBB, size);
    uct_mem_free(&mem);
}

/*
 * reproduce issue #1284, main thread is registering memory while another thread
 * allocates and releases memory.
 */
UCS_TEST_SKIP_COND_P(test_md, reg_multi_thread,
                     !check_reg_mem_type(UCS_MEMORY_TYPE_HOST)) {
    ucs_status_t status;
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

UCS_TEST_P(test_md, sockaddr_accessibility) {
    ucs_sock_addr_t sock_addr;
    struct ifaddrs *ifaddr, *ifa;

    /* currently we don't have MDs with deprecated capability */
    ASSERT_FALSE(check_caps(UCT_MD_FLAG_SOCKADDR));
    ASSERT_NE(NULL, uintptr_t(md()));
    ASSERT_TRUE(getifaddrs(&ifaddr) != -1);
    /* go through a linked list of available interfaces */
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (!ucs::is_interface_usable(ifa)) {
            continue;
        }

        sock_addr.addr = ifa->ifa_addr;
        UCS_TEST_MESSAGE << "Testing " << ifa->ifa_name << " with "
                         << ucs::sockaddr_to_str(ifa->ifa_addr);
        ASSERT_FALSE(uct_md_is_sockaddr_accessible(md(), &sock_addr,
                                                   UCT_SOCKADDR_ACC_LOCAL));
        ASSERT_FALSE(uct_md_is_sockaddr_accessible(md(), &sock_addr,
                                                   UCT_SOCKADDR_ACC_REMOTE));
    }
    freeifaddrs(ifaddr);
}

/* This test registers region N times and later deregs it N/2 times and
 * invalidates N/2 times - mix multiple dereg and invalidate calls.
 * Guarantee that all packed keys are unique. */
UCS_TEST_SKIP_COND_P(test_md, invalidate, !check_caps(UCT_MD_FLAG_INVALIDATE))
{
    static const size_t size       = 1 * UCS_MBYTE;
    const int limit                = 2000 / ucs::test_time_multiplier();
    static const unsigned md_flags = UCT_MD_MEM_ACCESS_REMOTE_PUT |
                                     UCT_MD_MEM_ACCESS_REMOTE_GET;
    std::vector<uct_mem_h> memhs;
    std::set<uint64_t> keys_set;
    uct_mem_h memh;
    void *ptr;
    size_t mem_reg_count; /* how many mem_reg operations to apply */
    size_t iter;
    ucs_status_t status;
    uct_md_mem_dereg_params_t dereg_params;
    uct_md_mkey_pack_params_t pack_params;
    uint64_t key;

    if (GetParam().md_name == "cuda_ipc") {
        UCS_TEST_SKIP_R("test not needed with cuda-ipc");
    }

    comp().comp.func         = dereg_cb;
    comp().comp.status       = UCS_OK;
    comp().self              = this;

    dereg_params.field_mask  = UCT_MD_MEM_DEREG_FIELD_FLAGS |
                               UCT_MD_MEM_DEREG_FIELD_MEMH |
                               UCT_MD_MEM_DEREG_FIELD_COMPLETION;
    dereg_params.comp        = &comp().comp;

    ptr                      = malloc(size);

    for (mem_reg_count = 1; mem_reg_count < limit; mem_reg_count++) {
        comp().comp.count = (mem_reg_count + 1) / 2;
        m_comp_count = 0;

        status = uct_md_mem_reg(md(), ptr, size, md_flags, &memh);
        ASSERT_UCS_OK(status);
        memhs.push_back(memh);

        pack_params.field_mask = UCT_MD_MKEY_PACK_FIELD_FLAGS;
        pack_params.flags      = UCT_MD_MKEY_PACK_FLAG_INVALIDATE;
        status = uct_md_mkey_pack_v2(md(), memh, &pack_params, &key);
        ASSERT_UCS_OK(status);

        bool is_unique = keys_set.insert(key).second;
        ASSERT_TRUE(is_unique) << keys_set.size()
                               << "-th key is not unique";

        for (iter = 1; iter < mem_reg_count; iter++) {
            status = uct_md_mem_reg(md(), ptr, size, md_flags, &memh);
            ASSERT_UCS_OK(status);
            memhs.push_back(memh);
        }

        for (iter = 0; iter < mem_reg_count; iter++) {
            /* mix dereg and dereg(invalidate) operations */
            ASSERT_EQ(0, m_comp_count);
            memh = memhs.back();
            if ((iter & 1) == 0) { /* on even iteration invalidate handle */
                dereg_params.flags = UCT_MD_MEM_DEREG_FLAG_INVALIDATE;
            } else {
                dereg_params.flags = 0;
            }

            dereg_params.memh = memh;
            status            = uct_md_mem_dereg_v2(md(), &dereg_params);
            ASSERT_UCS_OK(status);
            memhs.pop_back();
        }

        ASSERT_TRUE(memhs.empty());
        EXPECT_EQ(1, m_comp_count);
    }

    free(ptr);
}

UCS_TEST_SKIP_COND_P(test_md, dereg_bad_arg,
                     !check_reg_mem_type(UCS_MEMORY_TYPE_HOST) ||
                     !ENABLE_PARAMS_CHECK)
{
    static const size_t size       = 1 * UCS_MBYTE;
    static const unsigned md_flags = UCT_MD_MEM_ACCESS_REMOTE_PUT |
                                     UCT_MD_MEM_ACCESS_REMOTE_GET;
    uct_mem_h memh;
    void *ptr;
    ucs_status_t status;
    uct_md_mem_dereg_params_t params;

    ptr    = malloc(size);
    status = uct_md_mem_reg(md(), ptr, size, md_flags, &memh);
    ASSERT_UCS_OK(status);

    comp().comp.func   = dereg_cb;
    comp().comp.count  = 1;
    comp().comp.status = UCS_OK;
    comp().self        = this;
    params.memh        = memh;
    params.flags       = UCT_MD_MEM_DEREG_FLAG_INVALIDATE;
    params.comp        = &comp().comp;

    if (!check_caps(UCT_MD_FLAG_INVALIDATE)) {
        params.field_mask = UCT_MD_MEM_DEREG_FIELD_COMPLETION |
                            UCT_MD_MEM_DEREG_FIELD_FLAGS |
                            UCT_MD_MEM_DEREG_FIELD_MEMH;
        status            = uct_md_mem_dereg_v2(md(), &params);
        ASSERT_UCS_STATUS_EQ(UCS_ERR_UNSUPPORTED, status);

        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH;
        status            = uct_md_mem_dereg_v2(md(), &params);
    } else {
        params.field_mask = UCT_MD_MEM_DEREG_FIELD_COMPLETION;
        status            = uct_md_mem_dereg_v2(md(), &params);
        ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);

        params.field_mask = UCT_MD_MEM_DEREG_FIELD_COMPLETION |
                            UCT_MD_MEM_DEREG_FIELD_FLAGS;
        status            = uct_md_mem_dereg_v2(md(), &params);
        ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);

        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH |
                            UCT_MD_MEM_DEREG_FIELD_FLAGS;
        status            = uct_md_mem_dereg_v2(md(), &params);
        ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, status);

        params.field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH |
                            UCT_MD_MEM_DEREG_FIELD_COMPLETION |
                            UCT_MD_MEM_DEREG_FIELD_FLAGS;
        status            = uct_md_mem_dereg_v2(md(), &params);
    }

    EXPECT_UCS_OK(status);
    free(ptr);
}

UCT_MD_INSTANTIATE_TEST_CASE(test_md)

class test_md_fork : private ucs::clear_dontcopy_regions, public test_md {
};

UCS_TEST_SKIP_COND_P(test_md_fork, fork,
                     !check_reg_mem_type(UCS_MEMORY_TYPE_HOST),
                     "RCACHE_CHECK_PFN=1")
{
    static size_t REG_SIZE = 100;
    ucs_status_t status;
    int child_status;
    uct_mem_h memh;
    char *page = NULL;
    pid_t pid;

    EXPECT_EQ(0, posix_memalign((void **)&page, ucs_get_page_size(), REG_SIZE));
    memset(page, 42, REG_SIZE);

    status = uct_md_mem_reg(md(), page, REG_SIZE, UCT_MD_MEM_ACCESS_ALL, &memh);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    /* dereg can keep the region pinned in the registration cache */
    status = uct_md_mem_dereg(md(), memh);
    EXPECT_UCS_OK(status);

    pid = fork();
    if (pid == 0) {
        char buf[REG_SIZE];
        memset(buf, 42, REG_SIZE);
        /* child touch the page */
        EXPECT_EQ(0, memcmp(page, buf, REG_SIZE));
        exit(0);
    }

    EXPECT_NE(-1, pid);
    memset(page, 42, REG_SIZE);

    /* verify that rcache was flushed before fork()
     * PFN failure will be triggered otherwise */
    status = uct_md_mem_reg(md(), page, REG_SIZE, UCT_MD_MEM_ACCESS_ALL, &memh);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(memh != UCT_MEM_HANDLE_NULL);

    status = uct_md_mem_dereg(md(), memh);
    EXPECT_UCS_OK(status);

    ASSERT_EQ(pid, waitpid(pid, &child_status, 0));
    EXPECT_TRUE(WIFEXITED(child_status)) << ucs::exit_status_info(child_status);

    if (!RUNNING_ON_VALGRIND) {
        /* Under valgrind, leaks are possible due to early exit, so don't expect
         * an exit status of 0
         */
        EXPECT_EQ(0, WEXITSTATUS(child_status)) <<
                ucs::exit_status_info(child_status);
    }

    free(page);
}

UCT_MD_INSTANTIATE_TEST_CASE(test_md_fork)

