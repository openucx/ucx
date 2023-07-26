/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CMA_MD_H_
#define UCT_CMA_MD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/status.h>
#include <uct/base/uct_md.h>
#include <uct/api/v2/uct_v2.h>

#include <sys/types.h>
#include <unistd.h>

extern uct_component_t uct_cma_component;

ucs_status_t uct_cma_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr);

#endif
