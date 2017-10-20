/**
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef LIBPERF_ROCM_H
#define LIBPERF_ROCM_H

#if HAVE_ROCM

#include <ucs/sys/compiler.h>

BEGIN_C_DECLS


ucs_status_t rocm_init(ucx_perf_params_t *params);
void rocm_shutdown();

void *rocm_allocate_transfer_buffer(ucx_perf_params_t *params, size_t buffer_size);
void  rocm_free_transfer_buffer(void *p);


END_C_DECLS

#endif /*  HAVE_ROCM */


#endif /* LIBPERF_ROCM_H*/

