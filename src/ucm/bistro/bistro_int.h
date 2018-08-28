/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_BISTRO_BISTRO_INT_H_
#define UCM_BISTRO_BISTRO_INT_H_

#include <sys/mman.h>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>

#include <ucm/bistro/bistro.h>
#include <ucm/util/sys.h>
#include <ucm/util/log.h>
#include <ucs/sys/math.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>

#define UCM_PROT_READ_WRITE_EXEC (PROT_READ | PROT_WRITE | PROT_EXEC)
#define UCM_PROT_READ_EXEC       (PROT_READ | PROT_EXEC)

#define UCM_LOOKUP_SYMBOL(_func, _symbol) \
    _func = ucm_bistro_lookup(_symbol);   \
    if (!_func) {                         \
        return UCS_ERR_NO_ELEM;           \
    }

ucs_status_t ucm_bistro_apply_patch(void *dst, void *patch, size_t len);

ucs_status_t ucm_bistro_create_restore_point(void *addr, ucm_bistro_restore_point_t **rp);

static inline void *ucm_bistro_lookup(const char *symbol)
{
    void *addr;

    ucs_assert(symbol != NULL);

    addr = dlsym(RTLD_NEXT, symbol);
    if (!addr) {
        addr = dlsym(RTLD_DEFAULT, symbol);
    }
    return addr;
}

#endif
