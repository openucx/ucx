/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#if defined(__x86_64__)

#include <common/test.h>
extern "C" {
#include <ucs/sys/sys.h>
#include <ucs/arch/cpu.h>
#include <ucs/time/time.h>
}

#include <sys/mman.h>

class test_arch : public ucs::test {
protected:
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
};

UCS_TEST_SKIP_COND_F(test_arch, memcpy, RUNNING_ON_VALGRIND || !ucs::perf_retry_count) {
    const double diff      = 0.90; /* allow 10% fluctuations */
    const double timeout   = 30; /* 30 seconds to complete test successfully */
    double memcpy_bw       = 0;
    double memcpy_relax_bw = 0;
    double secs;
    size_t size;
    char memunits_str[256];
    char thresh_min_str[16];
    char thresh_max_str[16];
    int i;

    ucs_memunits_to_str(ucs_global_opts.arch.builtin_memcpy_min,
                        thresh_min_str, sizeof(thresh_min_str));
    ucs_memunits_to_str(ucs_global_opts.arch.builtin_memcpy_max,
                        thresh_max_str, sizeof(thresh_max_str));
    UCS_TEST_MESSAGE << "Using memcpy relaxed for size " <<
                        thresh_min_str << ".." <<
                        thresh_max_str;
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

#endif
