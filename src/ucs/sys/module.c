/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
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
#include <ucs/datastruct/string_buffer.h>
#include <ucs/datastruct/string_set.h>
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
#include <dirent.h>


#ifdef UCX_SHARED_LIB

#define UCS_MODULE_PATH_MEMTRACK_NAME   "module_path"
#define UCS_MODULE_SRCH_PATH_MAX        2

#define ucs_module_debug(_fmt, ...) \
    ucs_log(ucs_min(UCS_LOG_LEVEL_DEBUG, ucs_global_opts.module_log_level), \
            _fmt, ##  __VA_ARGS__)
#define ucs_module_trace(_fmt, ...) \
    ucs_log(ucs_min(UCS_LOG_LEVEL_TRACE, ucs_global_opts.module_log_level), \
            _fmt, ##  __VA_ARGS__)

static char *ucs_module_srch_paths_buf[UCS_MODULE_SRCH_PATH_MAX];

static struct {
    ucs_init_once_t              init;
    char                         module_ext[NAME_MAX];
    ucs_array_s(unsigned, char*) srch_path;
} ucs_module_loader_state = {
    .init       = UCS_INIT_ONCE_INITIALIZER,
    .module_ext = ".so", /* default extension */
    .srch_path  = UCS_ARRAY_FIXED_INITIALIZER(ucs_module_srch_paths_buf,
                                              UCS_MODULE_SRCH_PATH_MAX)
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
    *ucs_array_append(&ucs_module_loader_state.srch_path, goto out) = path;

out:
    ucs_free(dlpath_dup);
}

/* Should be called with lock held */
static void ucs_module_loader_add_install_dir()
{
    *ucs_array_append(&ucs_module_loader_state.srch_path,
                      return ) = ucs_global_opts.module_dir;
}

static void ucs_module_loader_init_paths()
{
    UCS_INIT_ONCE(&ucs_module_loader_state.init) {
        ucs_assert(ucs_array_length(&ucs_module_loader_state.srch_path) == 0);
        ucs_module_loader_add_dl_dir();
        ucs_module_loader_add_install_dir();
        ucs_assert(ucs_array_length(&ucs_module_loader_state.srch_path) <=
                   UCS_MODULE_SRCH_PATH_MAX);
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
    char *fullpath, *buffer;
    init_func_t init_func;
    ucs_status_t status;

    status = ucs_string_alloc_path_buffer(&buffer, "buffer");
    if (status != UCS_OK) {
        goto out;
    }

    fullpath = realpath(module_path, buffer);
    if (fullpath == NULL) {
        goto out_free_buffer;
    }

    ucs_module_trace("loaded %s [%p]", fullpath, dl);

    init_func = (init_func_t)ucs_module_dlsym_shallow(module_path, dl,
                                                      module_init_name);
    if (init_func == NULL) {
        goto out_free_buffer;
    }

    ucs_module_trace("calling '%s' in '%s': [%p]", module_init_name, fullpath,
                     init_func);
    status = init_func();
    if (status != UCS_OK) {
        ucs_module_debug("initializing '%s' failed: %s, unloading", fullpath,
                         ucs_status_string(status));
        dlclose(dl);
    }

out_free_buffer:
    ucs_free(buffer);
out:
    return;
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

static int ucs_module_flags_to_dlopen_mode(unsigned flags)
{
    int mode = RTLD_LAZY;

    if (flags & UCS_MODULE_LOAD_FLAG_NODELETE) {
        mode |= RTLD_NODELETE;
    }
    if (flags & UCS_MODULE_LOAD_FLAG_GLOBAL) {
        mode |= RTLD_GLOBAL;
    } else {
        mode |= RTLD_LOCAL;
    }

    return mode;
}

static void *ucs_module_try_load(const char *module_path, int mode)
{
    const char *error;
    void *dl;

    (void)dlerror();
    dl = dlopen(module_path, mode);
    if (dl == NULL) {
        error = dlerror();
        ucs_module_debug("dlopen('%s', mode=0x%x) failed: %s", module_path,
                         mode, error ? error : "Unknown error");
    }

    return dl;
}

static int ucs_module_filename_match(const char *filename, const char *prefix,
                                     size_t prefix_len)
{
    size_t suffix_len = strlen(ucs_module_loader_state.module_ext);
    size_t name_len   = strlen(filename);

    return (name_len > (prefix_len + suffix_len)) &&
           (strncmp(filename, prefix, prefix_len) == 0) &&
           (strcmp(filename + name_len - suffix_len,
                   ucs_module_loader_state.module_ext) == 0);
}

/* e.g. "libuct_ib_mlx5.so" -> "libuct_ib_mlx5" (strip suffix for set key) */
static void
ucs_module_filename_to_base(const char *filename, char *base, size_t base_max)
{
    size_t suffix_len = strlen(ucs_module_loader_state.module_ext);
    size_t name_len   = strlen(filename);
    size_t base_len   = name_len - suffix_len;

    if (base_len >= base_max) {
        base_len = base_max - 1;
    }
    memcpy(base, filename, base_len);
    base[base_len] = '\0';
}

static void ucs_module_load_from_dir(const char *dir, const char *framework,
                                     int mode, int check_enabled,
                                     ucs_string_set_t *loaded_set)
{
    char prefix[NAME_MAX];
    char base[NAME_MAX];
    char *module_path;
    size_t prefix_len;
    struct dirent *entry;
    ucs_status_t status;
    void *dl;
    DIR *dp;

    dp = opendir(dir);
    if (dp == NULL) {
        ucs_module_trace("cannot open module directory '%s'", dir);
        return;
    }

    status = ucs_string_alloc_path_buffer(&module_path, "module_path");
    if (status != UCS_OK) {
        goto out_closedir;
    }

    snprintf(prefix, sizeof(prefix), "lib%s_", framework);
    prefix_len = strlen(prefix);

    while ((entry = readdir(dp)) != NULL) {
        if (!ucs_module_filename_match(entry->d_name, prefix, prefix_len)) {
            continue;
        }

        ucs_module_filename_to_base(entry->d_name, base, sizeof(base));
        if (strchr(base + prefix_len, '_') != NULL) {
            ucs_module_debug("module name contains '_': %s, skipping", base + prefix_len);
            continue;
        }

        snprintf(module_path, PATH_MAX, "%s/%s", dir, entry->d_name);
        if (check_enabled && !ucs_module_is_enabled(base + prefix_len)) {
            ucs_module_debug("module is disabled: %s, skipping", base + prefix_len);
            continue;
        }

        if (ucs_string_set_contains(loaded_set, base)) {
            continue;
        }

        dl = dlopen(module_path, RTLD_LAZY | RTLD_NOLOAD);
        if (dl != NULL) {
            /* Already loaded (e.g. linked into the executable). Still run
             * init so sub-framework modules (e.g. uct_ib_mlx5 -> gda) load. */
            ucs_module_init(module_path, dl);
            (void)ucs_string_set_add(loaded_set, base);
            continue;
        }

        dl = ucs_module_try_load(module_path, mode);
        if (dl == NULL) {
            continue;
        }

        ucs_module_init(module_path, dl);
        (void)ucs_string_set_add(loaded_set, base);
    }

    ucs_free(module_path);
out_closedir:
    closedir(dp);
}

static void ucs_module_check_expected_loaded(const char *framework,
                                             const char *expected_modules,
                                             const ucs_string_set_t *loaded_set)
{
    ucs_string_buffer_t strb;
    char *module_name;
    char base[NAME_MAX];

    if (expected_modules == NULL || expected_modules[0] == '\0') {
        return;
    }

    ucs_string_buffer_init(&strb);
    ucs_string_buffer_appendf(&strb, "%s", expected_modules);

    ucs_string_buffer_for_each_token(module_name, &strb, ":") {
        if (ucs_module_is_enabled(module_name)) {
            snprintf(base, sizeof(base), "lib%s_%s", framework, module_name);
            if (!ucs_string_set_contains(loaded_set, base)) {
                ucs_module_debug("required module '%s' for framework '%s' "
                                 "was not loaded",
                                 module_name, framework);
            }
        }
    }

    ucs_string_buffer_cleanup(&strb);
}

#endif /* UCX_SHARED_LIB */

void ucs_load_modules(const char *framework, const char *expected_modules,
                      ucs_init_once_t *init_once, unsigned flags)
{
#ifdef UCX_SHARED_LIB
    unsigned i;
    int mode;
    ucs_string_set_t loaded_set;

    ucs_module_loader_init_paths();

    UCS_INIT_ONCE(init_once) {
        ucs_assert(ucs_sys_is_dynamic_lib());

        ucs_string_set_init(&loaded_set);
        mode = ucs_module_flags_to_dlopen_mode(flags);

        /* Load modules from directories */
        for (i = 0; i < ucs_global_opts.plugin_path.count; ++i) {
            ucs_module_load_from_dir(ucs_global_opts.plugin_path.names[i],
                                     framework, mode, 0, &loaded_set);
        }

        for (i = 0; i < ucs_array_length(&ucs_module_loader_state.srch_path);
             ++i) {
            ucs_module_load_from_dir(
                    ucs_array_elem(&ucs_module_loader_state.srch_path, i),
                    framework, mode, 1, &loaded_set);
        }

        ucs_module_check_expected_loaded(framework, expected_modules,
                                         &loaded_set);

        ucs_string_set_cleanup(&loaded_set);
    }
#endif /* UCX_SHARED_LIB */
}
