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


static inline void ucs_module_normalize_base(char *base, const char *prefix)
{
#ifdef UCX_MODULE_FILE_SUFFIX
    size_t prefix_len = strlen(prefix);
    size_t suffix_len = strlen(UCX_MODULE_FILE_SUFFIX);
    size_t base_len   = strlen(base);

    /*
     * Module files are first filtered by the lib<framework>_ prefix. Strip
     * the configured private file suffix only for bases that still match that
     * framework prefix after removing the shared-library extension.
     */
    if ((base_len > (prefix_len + suffix_len)) &&
        !strncmp(base, prefix, prefix_len) &&
        !strcmp(base + base_len - suffix_len, UCX_MODULE_FILE_SUFFIX)) {
        base[base_len - suffix_len] = '\0';
    }
#endif
}


#endif
