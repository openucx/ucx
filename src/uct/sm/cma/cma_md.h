/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CMA_MD_H_
#define UCT_CMA_MD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>
#include <uct/base/uct_md.h>

#include <sys/types.h>
#include <unistd.h>

extern uct_md_component_t uct_cma_md_component;

ucs_status_t uct_cma_md_query(uct_md_h md, uct_md_attr_t *md_attr);

#endif
