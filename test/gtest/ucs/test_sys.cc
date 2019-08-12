/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#define __STDC_LIMIT_MACROS // needed for SIZE_MAX

#include <common/test.h>
extern "C" {
#include <ucs/sys/module.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/sock.h>
#include <ucs/type/spinlock.h>
#include <ucs/time/time.h>
#include <ucs/arch/cpu.h>
}

#include <sys/mman.h>
#include <set>

class test_sys : public ucs::test {
protected:
    static int get_mem_prot(void *address, size_t size) {
        return ucs_get_mem_prot((uintptr_t)address, (uintptr_t)address + size);
    }

    void test_memunits(size_t size, const char *expected) {
        char buf[256];

        ucs_memunits_to_str(size, buf, sizeof(buf));
        EXPECT_EQ(std::string(expected), buf);
    }

    /* have to add wrapper for ucs_memcpy_relaxed because pure "C" inline call could
     * not be used as template argument */
    static inline void *memcpy_relaxed(void *dst, const void *src, size_t size)
    {
        return ucs_memcpy_relaxed(dst, src, size);
    }

    template <void* (C)(void*, const void*, size_t)>
    double measure_memcpy_bandwidth(size_t size)
    {
        ucs_time_t start_time, end_time;
        void *src, *dst;
        double result = 0.0;
        int iter;

        src = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (src == MAP_FAILED) {
            goto out;
        }

        dst = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (dst == MAP_FAILED) {
            goto out_unmap_src;
        }

        memset(dst, 0, size);
        memset(src, 0, size);
        memcpy(dst, src, size);

        iter = 0;
        start_time = ucs_get_time();
        do {
            C(dst, src, size);
            end_time = ucs_get_time();
            ++iter;
        } while (end_time < start_time + ucs_time_from_sec(0.5));

        result = size * iter / ucs_time_to_sec(end_time - start_time);

        munmap(dst, size);
    out_unmap_src:
        munmap(src, size);
    out:
        return result;
    }

    static void check_cache_type(ucs_cpu_cache_type_t type, const char *name)
    {
        size_t cache;
        char memunits[32];

        cache = ucs_cpu_get_cache_size(type);

        ucs_memunits_to_str(cache, memunits, sizeof(memunits));
        UCS_TEST_MESSAGE << name << " cache: " << memunits;
    }
};

UCS_TEST_F(test_sys, uuid) {
    std::set<uint64_t> uuids;
    for (unsigned i = 0; i < 10000; ++i) {
        uint64_t uuid = ucs_generate_uuid(0);
        std::pair<std::set<uint64_t>::iterator, bool> ret = uuids.insert(uuid);
        ASSERT_TRUE(ret.second);
    }
}

UCS_TEST_F(test_sys, machine_guid) {
    uint64_t guid1 = ucs_machine_guid();
    uint64_t guid2 = ucs_machine_guid();
    EXPECT_EQ(guid1, guid2);
}

UCS_TEST_F(test_sys, spinlock) {
    ucs_spinlock_t lock;
    pthread_t self;

    self = pthread_self();

    ucs_spinlock_init(&lock);

    ucs_spin_lock(&lock);
    EXPECT_TRUE(ucs_spin_is_owner(&lock, self));

    /* coverity[double_lock] */
    ucs_spin_lock(&lock);
    EXPECT_TRUE(ucs_spin_is_owner(&lock, self));

    ucs_spin_unlock(&lock);
    EXPECT_TRUE(ucs_spin_is_owner(&lock, self));

    /* coverity[double_unlock] */
    ucs_spin_unlock(&lock);
    EXPECT_FALSE(ucs_spin_is_owner(&lock, self));
}

UCS_TEST_F(test_sys, get_mem_prot) {
    int x;

    ASSERT_TRUE( get_mem_prot(&x, sizeof(x)) & PROT_READ );
    ASSERT_TRUE( get_mem_prot(&x, sizeof(x)) & PROT_WRITE );
    ASSERT_TRUE( get_mem_prot((void*)&get_mem_prot, 1) & PROT_EXEC );

    ucs_time_t start_time = ucs_get_time();
    get_mem_prot(&x, sizeof(x));
    ucs_time_t duration = ucs_get_time() - start_time;
    UCS_TEST_MESSAGE << "Time: " << ucs_time_to_usec(duration) << " us";
}

UCS_TEST_F(test_sys, fcntl) {
    ucs_status_t status;
    int fd, fl;

    fd = open("/dev/null", O_RDONLY);
    if (fd < 0) {
        FAIL();
    }

    /* Add */
    status = ucs_sys_fcntl_modfl(fd, O_NONBLOCK, 0);
    EXPECT_TRUE(status == UCS_OK);

    fl = fcntl(fd, F_GETFL);
    EXPECT_GE(fl, 0);
    EXPECT_TRUE(fl & O_NONBLOCK);

    /* Remove */
    status = ucs_sys_fcntl_modfl(fd, 0, O_NONBLOCK);
    EXPECT_TRUE(status == UCS_OK);

    fl = fcntl(fd, F_GETFL);
    EXPECT_GE(fl, 0);
    EXPECT_FALSE(fl & O_NONBLOCK);

    close(fd);
}

UCS_TEST_F(test_sys, memory) {
    size_t phys_size = ucs_get_phys_mem_size();
    UCS_TEST_MESSAGE << "Physical memory size: " << ucs::size_value(phys_size);
    EXPECT_GT(phys_size, 1ul * 1024 * 1024);
}

extern "C" {
int test_module_loaded = 0;
}

UCS_TEST_F(test_sys, module) {
    UCS_MODULE_FRAMEWORK_DECLARE(test);

    EXPECT_EQ(0, test_module_loaded);
    UCS_MODULE_FRAMEWORK_LOAD(test, 0);
    EXPECT_EQ(1, test_module_loaded);
}

UCS_TEST_F(test_sys, memunits_to_str) {
    test_memunits(256, "256");
    test_memunits(1256, "1256");
    test_memunits(UCS_KBYTE, "1K");
    test_memunits(UCS_MBYTE + UCS_KBYTE, "1025K");
    test_memunits(UCS_GBYTE, "1G");
    test_memunits(2 * UCS_GBYTE, "2G");
    test_memunits(UCS_TBYTE, "1T");
    test_memunits(UCS_TBYTE * 1024, "1024T");
}

UCS_TEST_SKIP_COND_F(test_sys, memcpy, RUNNING_ON_VALGRIND || !ucs::perf_retry_count) {
    const double diff      = 0.95; /* allow 5% fluctuations */
    const double timeout   = 30; /* 30 seconds to complete test successfully */
    double memcpy_bw       = 0;
    double memcpy_relax_bw = 0;
    double secs;
    size_t size;
    char memunits_str[256];
    int i;

    for (size = 4096; size <= 256 * UCS_MBYTE; size *= 2) {
        secs = ucs_get_accurate_time();
        for (i = 0; ucs_get_accurate_time() - secs < timeout; i++) {
            memcpy_bw       = measure_memcpy_bandwidth<memcpy>(size);
            memcpy_relax_bw = measure_memcpy_bandwidth<memcpy_relaxed>(size);
            if (memcpy_relax_bw / memcpy_bw >= diff) {
                break;
            }
            usleep(1000); /* allow other tasks to complete */
        }
        ucs_memunits_to_str(size, memunits_str, sizeof(memunits_str));
        UCS_TEST_MESSAGE << memunits_str <<
                            " memcpy: "             << (memcpy_bw / UCS_GBYTE) <<
                            "GB/s memcpy relaxed: " << (memcpy_relax_bw / UCS_GBYTE) <<
                            "GB/s iterations: "     << i + 1;
        EXPECT_GE(memcpy_relax_bw / memcpy_bw, diff);
    }
}

UCS_TEST_F(test_sys, cpu_cache) {
    check_cache_type(UCS_CPU_CACHE_L1d, "L1d");
    check_cache_type(UCS_CPU_CACHE_L1i, "L1i");
    check_cache_type(UCS_CPU_CACHE_L2, "L2");
    check_cache_type(UCS_CPU_CACHE_L3, "L3");
}
