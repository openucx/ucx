/**
 * Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDU_MD_H
#define UCT_RDU_MD_H

#include <uct/base/uct_md.h>

extern uct_component_t uct_rdu_component;

#define UCT_RDU_TL_NAME "rdu"

/**
 * RDU memory domain.
 */
typedef struct uct_rdu_md {
    struct uct_md super;
} uct_rdu_md_t;

/**
 * RDU memory domain configuration.
 */
typedef struct uct_rdu_md_config {
    uct_md_config_t super;
} uct_rdu_md_config_t;

#endif /* UCT_RDU_MD_H */
