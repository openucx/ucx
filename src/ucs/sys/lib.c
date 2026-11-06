/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021-2026. ALL RIGHTS RESERVED.
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

void *ucs_sys_get_lib_sym(const void *lib_addr, const char *sym_name,
                          const char *sym_version)
{
    Dl_info lib_info;
    Dl_info sym_info;
    const char *error;
    void *dl;
    void *sym;

    (void)dlerror();
    if (dladdr(lib_addr, &lib_info) == 0) {
        ucs_debug("could not identify the library containing %p", lib_addr);
        return NULL;
    }

    dl = dlopen(lib_info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    if (dl == NULL) {
        error = dlerror();
        ucs_debug("dlopen('%s') failed: %s", lib_info.dli_fname,
                  (error != NULL) ? error : "unknown error");
        return NULL;
    }

#if HAVE_DECL_DLVSYM
    /* The version a direct call would bind to when linked */
    sym = (sym_version != NULL) ? dlvsym(dl, sym_name, sym_version) :
                                  dlsym(dl, sym_name);
#else
    sym_version = NULL;
    sym         = dlsym(dl, sym_name);
#endif
    if (sym == NULL) {
        ucs_debug("%s@%s is not provided by %s", sym_name,
                  (sym_version != NULL) ? sym_version : "default",
                  lib_info.dli_fname);
    } else if ((dladdr(sym, &sym_info) == 0) ||
               (sym_info.dli_fbase != lib_info.dli_fbase)) {
        /* Found in a dependency, not in the requested library */
        ucs_debug("ignoring %s which is not defined in %s", sym_name,
                  lib_info.dli_fname);
        sym = NULL;
    }

    dlclose(dl);
    return sym;
}
