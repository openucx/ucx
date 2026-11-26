/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "lib.h"

#include <ucs/debug/log.h>


ucs_status_t ucs_sys_get_lib_info(Dl_info *dl_info)
{
    int ret;

    (void)dlerror();
    ret = dladdr(ucs_sys_get_lib_info, dl_info);
    if (ret == 0) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

const char *ucs_sys_get_lib_path()
{
    ucs_status_t status;
    Dl_info dl_info;

    status = ucs_sys_get_lib_info(&dl_info);
    if (status != UCS_OK) {
        return "<failed to resolve libucs path>";
    }

    return dl_info.dli_fname;
}

unsigned long ucs_sys_get_lib_base_addr()
{
    ucs_status_t status;
    Dl_info dl_info;

    status = ucs_sys_get_lib_info(&dl_info);
    if (status != UCS_OK) {
        return 0;
    }

    return (uintptr_t)dl_info.dli_fbase;
}

void *ucs_sys_lib_dlsym(const void *lib_addr, const char *symbol)
{
    Dl_info lib_info, sym_info;
    const char *error;
    void *dl_handle;
    void *sym;

    if (dladdr(lib_addr, &lib_info) == 0) {
        ucs_debug("no loaded library contains address %p", lib_addr);
        return NULL;
    }

    dl_handle = dlopen(lib_info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    if (dl_handle == NULL) {
        error = dlerror();
        ucs_debug("could not open loaded library %s: %s", lib_info.dli_fname,
                  (error != NULL) ? error : "unknown error");
        return NULL;
    }

    /* dlsym() also searches the dependencies of the library */
    (void)dlerror();
    sym = dlsym(dl_handle, symbol);
    if (sym == NULL) {
        error = dlerror();
        ucs_debug("%s is not found in %s: %s", symbol, lib_info.dli_fname,
                  (error != NULL) ? error : "unknown error");
    } else if ((dladdr(sym, &sym_info) == 0) ||
               (sym_info.dli_fbase != lib_info.dli_fbase)) {
        ucs_debug("%s at %p is not defined by %s", symbol, sym,
                  lib_info.dli_fname);
        sym = NULL;
    }

    /* The library stays loaded, since the caller keeps it loaded */
    dlclose(dl_handle);
    return sym;
}
