/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_MD_H
#define UCT_CUDA_MD_H

#include <uct/base/uct_md.h>

ucs_status_t uct_cuda_base_detect_memory_type(uct_md_h md, void *addr, size_t length,
                                              uct_memory_type_t *mem_type);

#endif
