/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_CONTIG_H_
#define UCP_DT_CONTIG_H_

#include "dt.h"

#include <ucp/core/ucp_mm.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>
#include <ucs/profile/profile.h>


#define UCP_DT_IS_CONTIG(_datatype) \
    (((_datatype)&UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_CONTIG)


static UCS_F_ALWAYS_INLINE size_t
ucp_contig_dt_elem_size(ucp_datatype_t datatype)
{
    return datatype >> UCP_DATATYPE_SHIFT;
}


static UCS_F_ALWAYS_INLINE size_t ucp_contig_dt_length(ucp_datatype_t datatype,
                                                       size_t count)
{
    ucs_assert(UCP_DT_IS_CONTIG(datatype));
    return count * ucp_contig_dt_elem_size(datatype);
}


static UCS_F_ALWAYS_INLINE void
ucp_dt_contig_pack(ucp_worker_h worker, void *dest, const void *src,
                   size_t length, ucs_memory_type_t mem_type, size_t total_len)
{
    if (ucs_likely(UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type))) {
        ucp_memcpy_pack(dest, src, length, total_len, "memcpy_pack");
    } else {
        ucp_mem_type_pack(worker, dest, src, length, mem_type);
    }
}


static UCS_F_ALWAYS_INLINE void
ucp_dt_contig_unpack(ucp_worker_h worker, void *dest, const void *src,
                     size_t length, ucs_memory_type_t mem_type, size_t total_len)
{
    if (ucs_likely(UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type))) {
        ucp_memcpy_unpack(dest, src, length, total_len, "memcpy_unpack");
    } else {
        ucp_mem_type_unpack(worker, dest, src, length, mem_type);
    }
}

#endif
