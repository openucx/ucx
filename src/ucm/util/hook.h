/**
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_UTIL_HOOK_H_
#define UCM_UTIL_HOOK_H_

#include <ucm/api/ucm.h>
#include <ucm/bistro/bistro.h>
#include <ucm/mmap/mmap.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>


typedef struct {
    ucm_reloc_patch_t patch;
    void              **orig_func_ptr;
} ucm_hook_func_t;


#define UCM_HOOK_FUNC_ENTRY(_func) \
    { \
        {#_func, ucm_override_##_func}, (void**)&ucm_orig_##_func \
    }


/* Install @a mode (bistro or reloc) for every loaded symbol in @a funcs.
 * Defined static in the header so it compiles into each loadable module
 * (libucm_cuda/libucm_rocm) and resolves the intercepted symbols in that
 * module's context, like ucm_reloc_get_orig(). */
static UCS_F_MAYBE_UNUSED ucs_status_t
ucm_install_hooks(ucm_hook_func_t *funcs, const char *name,
                  ucm_mmap_hook_mode_t mode, uint64_t hook_modes,
                  int *installed_hooks_p)
{
    ucm_hook_func_t *func;
    ucs_status_t status;
    void *func_ptr;
    int count;

    if (*installed_hooks_p & UCS_BIT(mode)) {
        return UCS_OK;
    }

    if (!(hook_modes & UCS_BIT(mode))) {
        ucm_debug("%s memory hooks mode %s is disabled", name,
                  ucm_mmap_hook_modes[mode]);
        return UCS_OK;
    }

    count = 0;
    for (func = funcs; func->patch.symbol != NULL; ++func) {
        func_ptr = ucm_reloc_get_orig(func->patch.symbol, func->patch.value);
        if (func_ptr == NULL) {
            continue;
        }

        if (mode == UCM_MMAP_HOOK_BISTRO) {
            status = ucm_bistro_patch(func_ptr, func->patch.value,
                                      func->patch.symbol, func->orig_func_ptr,
                                      NULL);
        } else if (mode == UCM_MMAP_HOOK_RELOC) {
            status = ucm_reloc_modify(&func->patch);
        } else {
            break;
        }

        if (status != UCS_OK) {
            ucm_diag("failed to install %s hook for '%s'",
                     ucm_mmap_hook_modes[mode], func->patch.symbol);
            return status;
        }

        ucm_debug("installed %s hook for '%s'", ucm_mmap_hook_modes[mode],
                  func->patch.symbol);
        ++count;
    }

    *installed_hooks_p |= UCS_BIT(mode);
    ucm_info("%s memory hooks mode %s: installed %d hooks", name,
             ucm_mmap_hook_modes[mode], count);
    return UCS_OK;
}

#endif
