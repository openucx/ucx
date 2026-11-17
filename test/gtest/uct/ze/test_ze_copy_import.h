/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TEST_ZE_COPY_IMPORT_H
#define UCT_TEST_ZE_COPY_IMPORT_H

#include <common/test_helpers.h>

extern "C" {
#include <uct/ze/copy/ze_copy_md.h>
}


/* Host pointer import entry points; skips the test when unavailable */
static inline const uct_ze_copy_host_import_t *ze_copy_host_import_or_skip()
{
    const uct_ze_copy_host_import_t *hi = uct_ze_copy_md_get_host_import();

    if (hi == NULL) {
        UCS_TEST_SKIP_R("Level Zero host pointer import is not available");
    }

    return hi;
}

#endif
