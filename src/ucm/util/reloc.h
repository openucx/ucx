/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_UTIL_RELOC_H_
#define UCM_UTIL_RELOC_H_

#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>


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
 * @param [in]  symbol       Symbol name,
 * @param [in]  replacement  Symbol replacement, which should be ignored.
 *
 * @return Original function pointer for 'symbol'.
 */
void* ucm_reloc_get_orig(const char *symbol, void *replacement);


#endif
