/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_MODULE_INT_H_
#define UCS_MODULE_INT_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <string.h>


static inline void ucs_module_normalize_base(char *base)
{
#ifdef UCX_MODULE_FILE_SUFFIX
    size_t suffix_len = strlen(UCX_MODULE_FILE_SUFFIX);
    size_t base_len   = strlen(base);

    /* Expected input after stripping .so is lib<framework>_<module>-<suffix>. */
    if ((base_len > suffix_len) &&
        !strcmp(base + base_len - suffix_len, UCX_MODULE_FILE_SUFFIX)) {
        base[base_len - suffix_len] = '\0';
    }
#endif
}


#endif
