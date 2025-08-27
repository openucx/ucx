/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef GDAKI_MEM_H_
#define GDAKI_MEM_H_

#include <ucs/sys/compiler_def.h>
#include <tools/perf/lib/libperf_int.h>


class gdaki_mem {
public:
    gdaki_mem(size_t size);
    ~gdaki_mem();

    void *get_cpu_ptr() const { return m_cpu_ptr; }
    void *get_gpu_ptr() const { return m_gpu_ptr; }

private:
    void   *m_gpu_ptr;
    void   *m_cpu_ptr;
    size_t m_size;
};

#endif /* GDAKI_MEM_H_ */
