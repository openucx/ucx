/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCPCM_MD_H_
#define UCT_TCPCM_MD_H_

#include "tcpcm_def.h"
#include <uct/base/uct_md.h>
#include <ucs/sys/sock.h>
#include <ucs/time/time.h>

/**
 * TCPCM memory domain.
 */
typedef struct uct_tcpcm_md {
    uct_md_t                 super;
    double                   addr_resolve_timeout; // FIXME
} uct_tcpcm_md_t;

/**
 * TCPCM memory domain configuration.
 */
typedef struct uct_tcpcm_md_config {
    uct_md_config_t          super;
    double                   addr_resolve_timeout; // FIXME
} uct_tcpcm_md_config_t;

extern uct_md_component_t uct_tcpcm_mdc;

ucs_status_t uct_tcpcm_md_query(uct_md_h md, uct_md_attr_t *md_attr);

int uct_tcpcm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                     uct_sockaddr_accessibility_t mode);

#endif
