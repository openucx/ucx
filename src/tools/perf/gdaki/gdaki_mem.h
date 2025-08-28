/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef GDAKI_MEM_H_
#define GDAKI_MEM_H_

#include <ucs/sys/compiler_def.h>
#include <tools/perf/lib/libperf_int.h>

BEGIN_C_DECLS

typedef struct {
    void   *gpu_ptr;
    void   *cpu_ptr;
    size_t size;
} gdaki_mem_t;

ucs_status_t gdaki_mem_create(gdaki_mem_t *mem, size_t size);
void gdaki_mem_destroy(gdaki_mem_t *mem);

END_C_DECLS

#endif /* GDAKI_MEM_H_ */
