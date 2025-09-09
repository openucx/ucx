/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_TESTS_H
#define UCT_TESTS_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf_int.h"

template<typename PSN>
class uct_perf_test_runner_base {
public:
    using psn_t = PSN;

    uct_perf_test_runner_base(ucx_perf_context_t &perf) :
        m_perf(perf)
    {}

    UCS_F_ALWAYS_INLINE static PSN *sn_ptr(void *buffer, size_t length)
    {
        return (PSN*)UCS_PTR_BYTE_OFFSET(buffer, length - sizeof(PSN));
    }

    UCS_F_ALWAYS_INLINE static void
    set_sn(void *dst_sn, ucs_memory_type_t dst_mem_type, const void *src_sn,
           const ucx_perf_allocator_t *allocator)
    {
        if (ucs_likely(allocator->mem_type == UCS_MEMORY_TYPE_HOST)) {
            ucs_assert(dst_mem_type == UCS_MEMORY_TYPE_HOST);
            *reinterpret_cast<PSN*>(dst_sn) = *reinterpret_cast<const PSN*>(src_sn);
        }

        allocator->memcpy(dst_sn, dst_mem_type, src_sn, UCS_MEMORY_TYPE_HOST,
                          sizeof(PSN));
    }

    UCS_F_ALWAYS_INLINE static PSN
    get_sn(const volatile void *sn, ucs_memory_type_t mem_type,
           const ucx_perf_allocator_t *allocator)
    {
        if (ucs_likely(mem_type == UCS_MEMORY_TYPE_HOST)) {
            return *reinterpret_cast<const volatile PSN*>(sn);
        }

        PSN host_sn;
        allocator->memcpy(&host_sn, UCS_MEMORY_TYPE_HOST,
                          const_cast<const void*>(sn), mem_type, sizeof(PSN));
        return host_sn;
    }

protected:
    ucx_perf_context_t &m_perf;
};

#endif /* UCT_TESTS_H */
