/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
* Copyright (C) Rivos Inc. 2023
* Copyright (C) Dandan Zhang, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_LOONGARCH64_CPU_H_
#define UCS_ARCH_LOONGARCH64_CPU_H_

#include <ucs/arch/generic/cpu.h>
#include <ucs/config/global_opts.h>
#include <ucs/config/types.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/compiler_def.h>

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <unistd.h>

BEGIN_C_DECLS

/** @file cpu.h */

#define UCS_ARCH_CACHE_LINE_SIZE 64

#define ucs_loongarch64_dbar(hint)   asm volatile ("dbar %0 " : : "I"(hint) : "memory")

#define crwrw           0b00000
#define cr_r_           0b00101
#define c_w_w           0b01010

#define orwrw           0b10000
#define or_r_           0b10101
#define o_w_w           0b11010

#define orw_w           0b10010
#define or_rw           0b10100

#define ucs_memory_bus_store_fence() ucs_loongarch64_dbar(c_w_w) 
#define ucs_memory_bus_load_fence()  ucs_loongarch64_dbar(cr_r_) 


#define ucs_memory_cpu_fence()               ucs_loongarch64_dbar(orwrw)     
#define ucs_memory_bus_cacheline_wc_flush()  ucs_memory_cpu_fence()
#define ucs_memory_cpu_store_fence()         ucs_loongarch64_dbar(o_w_w)      
#define ucs_memory_cpu_load_fence()          ucs_loongarch64_dbar(or_r_)     
#define ucs_memory_cpu_wc_fence()            ucs_memory_cpu_fence()

static inline double ucs_arch_get_clocks_per_sec()
{
    return ucs_arch_generic_get_clocks_per_sec();
}

static inline ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    return UCS_CPU_MODEL_LOONGARCH64;
}

static inline int ucs_arch_get_cpu_flag()
{
    return UCS_CPU_FLAG_UNKNOWN;
}

static inline void ucs_cpu_init()
{
}

ucs_cpu_vendor_t ucs_arch_get_cpu_vendor();

static inline ucs_status_t ucs_arch_get_cache_size(size_t *cache_sizes)
{
    return UCS_ERR_UNSUPPORTED;
}

static inline uint64_t ucs_arch_read_hres_clock()
{
    uint64_t cnt_id, time;
    __asm__ __volatile__ (
	"rdtime.d %0, %1\n\t"
	:"=&r"(time), "=&r"(cnt_id)
    );
    return time;
}

#define ucs_arch_wait_mem ucs_arch_generic_wait_mem

#if !HAVE___CLEAR_CACHE
static inline void ucs_arch_clear_cache(void *start, void *end)
{
      usc_memory_cpu_fence();
}
#endif

static inline void *ucs_memcpy_relaxed(void *dst, const void *src, size_t len,
                                       ucs_arch_memcpy_hint_t hint,
                                       size_t total_len)
{
    return memcpy(dst, src, len);
}

static UCS_F_ALWAYS_INLINE void
ucs_memcpy_nontemporal(void *dst, const void *src, size_t len)
{
    memcpy(dst, src, len);
}

END_C_DECLS

#endif

