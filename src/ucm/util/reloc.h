/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_UTIL_RELOC_H_
#define UCM_UTIL_RELOC_H_

#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <ucm/util/log.h>
#include <dlfcn.h>


/**
 * Tracks which symbols need to be patched for currently loaded libraries and
 * for libraries to be loaded in the future. We have the 'list' field so the
 * library could put those on a list without extra memory allocations.
 */
typedef struct ucm_reloc_patch {
    const char       *symbol;
    void             *value;
    void             *prev_value;
    ucs_list_link_t  list;
    char             **blacklist;
} ucm_reloc_patch_t;


/**
 * Modify process' relocation table.
 *
 * @param [in]  patch     What and how to modify. After this call, the structure
 *                         will be owned by the library and the same patching will
 *                         happen in all objects loaded subsequently.
 */
ucs_status_t ucm_reloc_modify(ucm_reloc_patch_t* patch);


/**
 * Get the original implementation of 'symbol', which is not equal to 'replacement'.
 *
 * This function is static to make sure the symbol search is done from the context
 * of the shared object which defines the replacement function.
 * If the replacement function is defined in a loadbale module, the symbols it
 * imports from other libraries may not be available in global scope.
 *
 * @param [in]  symbol       Symbol name.
 * @param [in]  replacement  Symbol replacement, which should be ignored.
 *
 * @return Original function pointer for 'symbol'.
 */
static void* UCS_F_MAYBE_UNUSED
ucm_reloc_get_orig(const char *symbol, void *replacement)
{
    const char *error;
    void *func_ptr;

    func_ptr = dlsym(RTLD_NEXT, symbol);
    if (func_ptr == NULL) {
        (void)dlerror();
        func_ptr = dlsym(RTLD_DEFAULT, symbol);
        if (func_ptr == replacement) {
            error = dlerror();
            ucm_fatal("could not find address of original %s(): %s", symbol,
                      error ? error : "Unknown error");
        }
    }

    ucm_debug("original %s() is at %p", symbol, func_ptr);
    return func_ptr;
}

#endif
