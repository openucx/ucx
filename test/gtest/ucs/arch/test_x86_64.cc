/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
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
        return ucs_memcpy_relaxed(dst, src, size, UCS_ARCH_MEMCPY_NT_NONE, size);
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

    void nt_buffer_transfer_test(ucs_arch_memcpy_hint_t hint) {
#ifndef __AVX__
        UCS_TEST_SKIP_R("Built without AVX support");
#else
        int i, j, result;
        char *test_window_src, *test_window_dst, *src, *dst, *dup;
        size_t len, total_size, test_window_size, hole_size, align;

        align            = 64;
        test_window_size = 8 * 1024;
        hole_size        = 2 * align;

        /*
         * Allocate a hole above and below the test_window_size
         * to check for writes beyond the designated area.
         */
        total_size = test_window_size + (2 * hole_size);

        posix_memalign((void **)&test_window_src, align, total_size);
        posix_memalign((void **)&test_window_dst, align, total_size);
        posix_memalign((void **)&dup, align, total_size);

        src = test_window_src + hole_size;
        dst = test_window_dst + hole_size;

        /* Initialize the regions with known patterns */
        memset(dup, 0x0, total_size);
        memset(test_window_src, 0xdeaddead, total_size);
        memset(test_window_dst, 0x0, total_size);

        len = 0;

        while (len < test_window_size) {
            for (i = 0; i < align; i++) {
                for (j = 0; j < align; j++) {
                    /* Perform the transfer */
                    ucs_x86_nt_buffer_transfer(dst + i, src + j, len, hint, len);
                    result = memcmp(src + j, dst + i, len);
                    EXPECT_EQ(0, result);

                    /* reset the copied region back to zero */
                    memset(dst + i, 0x0, len);

                    /* check for any modifications in the holes */
                    result = memcmp(test_window_dst, dup, total_size);
                    EXPECT_EQ(0, result);
                }
            }
            /* Check for each len for less than 1k sizes
             * Above 1k test steps of 53
             */
            if (len < 1024) {
                len++;
            } else {
                len += 53;
            }
        }

        free(test_window_src);
        free(test_window_dst);
        free(dup);
#endif
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

UCS_TEST_F(test_arch, nt_buffer_transfer_nt_src) {
    nt_buffer_transfer_test(UCS_ARCH_MEMCPY_NT_SOURCE);
}

UCS_TEST_F(test_arch, nt_buffer_transfer_nt_dst) {
    nt_buffer_transfer_test(UCS_ARCH_MEMCPY_NT_DEST);
}

UCS_TEST_F(test_arch, nt_buffer_transfer_nt_src_dst) {
    /* Make nt_dest_threshold zero to test the combination of hints */
    ucs_global_opts.arch.nt_dest_threshold = 0;
    nt_buffer_transfer_test(UCS_ARCH_MEMCPY_NT_SOURCE);
}
#endif
