/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SOCKCM_MD_H_
#define UCT_SOCKCM_MD_H_

#include "sockcm_def.h"
#include <uct/base/uct_md.h>
#include <ucs/sys/sock.h>
#include <ucs/time/time.h>

/*
 * SOCKCM memory domain.
 */
typedef struct uct_sockcm_md {
    uct_md_t                 super;
} uct_sockcm_md_t;

/*
 * SOCKCM memory domain configuration.
 */
typedef struct uct_sockcm_md_config {
    uct_md_config_t          super;
} uct_sockcm_md_config_t;

extern uct_md_component_t uct_sockcm_mdc;

ucs_status_t uct_sockcm_md_query(uct_md_h md, uct_md_attr_t *md_attr);

int uct_sockcm_is_sockaddr_accessible(uct_md_h md,
                                      const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode);

#endif
