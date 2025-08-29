/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef DEVICE_MEM_H_
#define DEVICE_MEM_H_

#include <ucs/sys/compiler_def.h>
#include <tools/perf/lib/libperf_int.h>

BEGIN_C_DECLS

typedef struct {
    void   *gpu_ptr;
    void   *cpu_ptr;
    size_t size;
} device_mem_t;

ucs_status_t device_mem_create(device_mem_t *mem, size_t size);
void device_mem_destroy(device_mem_t *mem);

END_C_DECLS

#endif /* DEVICE_MEM_H_ */
