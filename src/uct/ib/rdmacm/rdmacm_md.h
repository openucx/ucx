/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDMACM_MD_H_
#define UCT_RDMACM_MD_H_

#include "rdmacm_def.h"
#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>
#include <ucs/time/time.h>
#include <rdma/rdma_cma.h>

/**
 * RDMACM memory domain.
 */
typedef struct uct_rdmacm_md {
    uct_md_t                 super;
    double                   addr_resolve_timeout;
} uct_rdmacm_md_t;

/**
 * RDMACM memory domain configuration.
 */
typedef struct uct_rdmacm_md_config {
    uct_md_config_t          super;
    double                   addr_resolve_timeout;
} uct_rdmacm_md_config_t;

extern uct_md_component_t uct_rdmacm_mdc;

ucs_status_t uct_rdmacm_md_query(uct_md_h md, uct_md_attr_t *md_attr);

int uct_rdmacm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode);

#endif
