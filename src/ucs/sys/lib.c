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
#include <ucs/sys/string.h>
#include <link.h>


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

void *ucs_sys_dlsym_shallow(const char *lib_path, void *dl,
                            const char *symbol, ucs_log_level_t log_level)
{
    struct link_map *lm_entry;
    Dl_info dl_info;
    void *addr;
    int ret;

    addr = dlsym(dl, symbol);
    if (addr == NULL) {
        return NULL;
    }

    (void)dlerror();
    ret = dladdr(addr, &dl_info);
    if (ret == 0) {
        ucs_log(log_level, "dladdr(%p) [%s] failed: %s", addr, symbol,
                dlerror());
        return NULL;
    }

    (void)dlerror();
    ret = dlinfo(dl, RTLD_DI_LINKMAP, &lm_entry);
    if (ret) {
        ucs_log(log_level, "dlinfo(%p) [%s] failed: %s", dl, lib_path,
                dlerror());
        return NULL;
    }

    /* return the symbol only if it was found in the requested library, and not,
     * for example, in one of its dependencies.
     */
    if (lm_entry->l_addr != (uintptr_t)dl_info.dli_fbase) {
        ucs_log(log_level,
                "ignoring '%s' (%p) from %s (%p), expected in %s (%lx)",
                symbol, addr, ucs_basename(dl_info.dli_fname),
                dl_info.dli_fbase, ucs_basename(lib_path), lm_entry->l_addr);
        return NULL;
    }

    return addr;
}

void *ucs_sys_dlsym(void *dl, const char *sym_name)
{
    const char *error;
    void *sym;

    sym = dlsym(dl, sym_name);
    if (sym == NULL) {
        error = dlerror();
        ucs_debug("dlsym('%s') failed: %s", sym_name,
                  (error != NULL) ? error : "unknown error");
    }

    return sym;
}

void *ucs_sys_get_lib_sym(const void *lib_addr, const char *sym_name)
{
    Dl_info lib_info;
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

    sym = ucs_sys_dlsym_shallow(lib_info.dli_fname, dl, sym_name,
                                UCS_LOG_LEVEL_DEBUG);
    if (sym == NULL) {
        (void)dlerror();
        ucs_debug("%s is not provided by %s", sym_name, lib_info.dli_fname);
    }

    dlclose(dl);
    return sym;
}
