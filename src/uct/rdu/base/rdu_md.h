/**
 * Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDU_MD_H
#define UCT_RDU_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/topo/base/topo.h>

extern uct_component_t uct_rdu_component;

#define UCT_RDU_TL_NAME "rdu"

/* Maximal number of RDUs on a host */
#define UCT_RDU_MAX_RDUS 16

/**
 * RDU memory domain.
 */
typedef struct uct_rdu_md {
    struct uct_md    super;
    /* System device of each RDU, UCS_SYS_DEVICE_ID_UNKNOWN until resolved */
    ucs_sys_device_t sys_dev[UCT_RDU_MAX_RDUS];
    /* Whether an RDU beyond UCT_RDU_MAX_RDUS was reported */
    int              max_rdus_warned;
} uct_rdu_md_t;

/**
 * RDU memory domain configuration.
 */
typedef struct uct_rdu_md_config {
    uct_md_config_t super;
} uct_rdu_md_config_t;

#endif /* UCT_RDU_MD_H */
