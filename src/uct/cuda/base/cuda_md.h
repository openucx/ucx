/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_MD_H
#define UCT_CUDA_MD_H

#include <uct/base/uct_md.h>

int uct_cuda_is_mem_type_owned(uct_md_h md, void *addr, size_t length);

#endif
