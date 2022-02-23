/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE /* for dladdr(3) */
#endif

#include "module.h"

#include <ucs/sys/preprocessor.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <string.h>
#include <limits.h>
#include <dlfcn.h>
#include <link.h>
#include <libgen.h>


#ifdef UCX_SHARED_LIB

#define UCS_MODULE_PATH_MEMTRACK_NAME   "module_path"
#define UCS_MODULE_SRCH_PATH_MAX        2

#define ucs_module_debug(_fmt, ...) \
    ucs_log(ucs_min(UCS_LOG_LEVEL_DEBUG, ucs_global_opts.module_log_level), \
            _fmt, ##  __VA_ARGS__)
#define ucs_module_trace(_fmt, ...) \
    ucs_log(ucs_min(UCS_LOG_LEVEL_TRACE, ucs_global_opts.module_log_level), \
            _fmt, ##  __VA_ARGS__)

static struct {
    ucs_init_once_t  init;
    char             module_ext[NAME_MAX];
    unsigned         srchpath_cnt;
    char             *srch_path[UCS_MODULE_SRCH_PATH_MAX];
} ucs_module_loader_state = {
    .init         = UCS_INIT_ONCE_INITIALIZER,
    .module_ext   = ".so", /* default extension */
    .srchpath_cnt = 0,
    .srch_path    = { NULL, NULL}
};

/* Should be called with lock held */
static void ucs_module_loader_add_dl_dir()
{
    char *dlpath_dup = NULL;
    size_t max_length;
    Dl_info dl_info;
    const char *p;
    char *path;
    int ret;

    (void)dlerror();
    ret = dladdr((void*)&ucs_module_loader_state, &dl_info);
    if (ret == 0) {
        ucs_error("dladdr failed: %s", dlerror());
        return;
    }

    ucs_module_debug("ucs library path: %s", dl_info.dli_fname);

    /* copy extension */
    dlpath_dup = ucs_strdup(dl_info.dli_fname,
                            UCS_MODULE_PATH_MEMTRACK_NAME);
    if (dlpath_dup == NULL) {
        return;
    }

    p = ucs_basename(dlpath_dup);
    p = strchr(p, '.');
    if (p != NULL) {
        strncpy(ucs_module_loader_state.module_ext, p,
                sizeof(ucs_module_loader_state.module_ext) - 1);
    }
    ucs_free(dlpath_dup);

    /* copy directory component */
    dlpath_dup = ucs_strdup(dl_info.dli_fname,
                            UCS_MODULE_PATH_MEMTRACK_NAME);
    if (dlpath_dup == NULL) {
        return;
    }

    /* construct module directory path */
    max_length = strlen(dlpath_dup) +         /* directory */
                 1 +                          /* '/' */
                 strlen(UCX_MODULE_SUBDIR) +  /* sub-directory */
                 1;                           /* '\0' */
    path = ucs_malloc(max_length, UCS_MODULE_PATH_MEMTRACK_NAME);
    if (path == NULL) {
        goto out;
    }

    snprintf(path, max_length, "%s/%s", dirname(dlpath_dup), UCX_MODULE_SUBDIR);
    ucs_module_loader_state.srch_path[ucs_module_loader_state.srchpath_cnt++] = path;

out:
    ucs_free(dlpath_dup);
}

/* Should be called with lock held */
static void ucs_module_loader_add_install_dir()
{
    ucs_module_loader_state.srch_path[ucs_module_loader_state.srchpath_cnt++] =
                    ucs_global_opts.module_dir;
}

static void ucs_module_loader_init_paths()
{
    UCS_INIT_ONCE(&ucs_module_loader_state.init) {
        ucs_assert(ucs_module_loader_state.srchpath_cnt == 0);
        ucs_module_loader_add_dl_dir();
        ucs_module_loader_add_install_dir();
        ucs_assert(ucs_module_loader_state.srchpath_cnt <= UCS_MODULE_SRCH_PATH_MAX);
    }
}

/* Perform shallow search for a symbol */
static void *ucs_module_dlsym_shallow(const char *module_path, void *dl,
                                      const char *symbol)
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
        ucs_module_debug("dladdr(%p) [%s] failed: %s", addr, symbol, dlerror());
        return NULL;
    }

    (void)dlerror();
    ret = dlinfo(dl, RTLD_DI_LINKMAP, &lm_entry);
    if (ret) {
        ucs_module_debug("dlinfo(%p) [%s] failed: %s", dl, module_path, dlerror());
        return NULL;
    }

    /* return the symbol only if it was found in the requested library, and not,
     * for example, in one of its dependencies.
     */
    if (lm_entry->l_addr != (uintptr_t)dl_info.dli_fbase) {
        ucs_module_debug("ignoring '%s' (%p) from %s (%p), expected in %s (%lx)",
                         symbol, addr, ucs_basename(dl_info.dli_fname),
                         dl_info.dli_fbase, ucs_basename(module_path),
                         lm_entry->l_addr);
        return NULL;
    }

    return addr;
}

static void ucs_module_init(const char *module_path, void *dl)
{
    typedef ucs_status_t (*init_func_t)();

    const char *module_init_name =
                    UCS_PP_MAKE_STRING(UCS_MODULE_CONSTRUCTOR_NAME);
    char *fullpath, buffer[PATH_MAX];
    init_func_t init_func;
    ucs_status_t status;

    fullpath = realpath(module_path, buffer);
    ucs_module_trace("loaded %s [%p]", fullpath, dl);

    init_func = (init_func_t)ucs_module_dlsym_shallow(module_path, dl,
                                                      module_init_name);
    if (init_func == NULL) {
        return;
    }

    ucs_module_trace("calling '%s' in '%s': [%p]", module_init_name, fullpath,
                     init_func);
    status = init_func();
    if (status != UCS_OK) {
        ucs_module_debug("initializing '%s' failed: %s, unloading", fullpath,
                         ucs_status_string(status));
        dlclose(dl);
    }
}


static int ucs_module_is_enabled(const char *module_name)
{
    ucs_config_allow_list_mode_t mode = ucs_global_opts.modules.mode;
    int found;

    if (mode == UCS_CONFIG_ALLOW_LIST_ALLOW_ALL) {
        return 1;
    }

    found = ucs_config_names_search(&ucs_global_opts.modules.array,
                                    module_name) >= 0;
    return ((mode == UCS_CONFIG_ALLOW_LIST_ALLOW) && found) ||
           ((mode == UCS_CONFIG_ALLOW_LIST_NEGATE) && !found);
}

static void ucs_module_load_one(const char *framework, const char *module_name,
                                unsigned flags)
{
    char module_path[PATH_MAX] = {0};
    const char *error;
    unsigned i;
    void *dl;
    int mode;

    if (!ucs_module_is_enabled(module_name)) {
        ucs_module_trace("module '%s' is disabled by configuration",
                         module_name);
        return;
    }

    mode = RTLD_LAZY;
    if (flags & UCS_MODULE_LOAD_FLAG_NODELETE) {
        mode |= RTLD_NODELETE;
    }
    if (flags & UCS_MODULE_LOAD_FLAG_GLOBAL) {
        mode |= RTLD_GLOBAL;
    } else {
        mode |= RTLD_LOCAL;
    }

    ucs_module_trace("loading module '%s' with mode 0x%x", module_name, mode);

    for (i = 0; i < ucs_module_loader_state.srchpath_cnt; ++i) {
        snprintf(module_path, sizeof(module_path) - 1, "%s/lib%s_%s%s",
                 ucs_module_loader_state.srch_path[i], framework, module_name,
                 ucs_module_loader_state.module_ext);

        /* Clear error state */
        (void)dlerror();
        dl = dlopen(module_path, mode);
        if (dl != NULL) {
            ucs_module_init(module_path, dl);
            break;
        } else {
            /* If a module fails to load, silently give up */
            error = dlerror();
            ucs_module_debug("dlopen('%s', mode=0x%x) failed: %s", module_path,
                             mode, error ? error : "Unknown error");
        }
    }

    /* coverity[leaked_storage] : a loaded module is never unloaded */
}
#endif /* UCX_SHARED_LIB */

void ucs_load_modules(const char *framework, const char *modules,
                      ucs_init_once_t *init_once, unsigned flags)
{
#ifdef UCX_SHARED_LIB
    char *modules_str;
    char *saveptr;
    char *module_name;

    ucs_module_loader_init_paths();

    UCS_INIT_ONCE(init_once) {
        ucs_assert(ucs_sys_is_dynamic_lib());

        ucs_module_debug("loading modules for %s", framework);
        modules_str = ucs_strdup(modules, "modules_list");
        if (modules_str != NULL) {
            saveptr     = NULL;
            module_name = strtok_r(modules_str, ":", &saveptr);
            while (module_name != NULL) {
                ucs_module_load_one(framework, module_name, flags);
                module_name = strtok_r(NULL, ":", &saveptr);
            }
            ucs_free(modules_str);
        } else {
            ucs_error("failed to allocate module names list");
        }
    }
#endif /* UCX_SHARED_LIB */
}
