/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE /* for dladdr(3) */
#endif

#include "plugin.h"

#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/type/init_once.h>
#include <ucs/datastruct/list.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <dlfcn.h>

#ifdef UCX_SHARED_LIB

/* Plugin registry - stores loaded plugin per component */
typedef struct ucs_plugin_registry_entry {
    ucs_list_link_t list;
    char *component;
    ucs_plugin_desc_t plugin;
} ucs_plugin_registry_entry_t;

static ucs_init_once_t ucs_plugin_registry_init_once = UCS_INIT_ONCE_INITIALIZER;
static UCS_LIST_HEAD(ucs_plugin_registry);

static ucs_plugin_registry_entry_t*
ucs_plugin_registry_find(const char *component)
{
    ucs_plugin_registry_entry_t *entry;

    ucs_list_for_each(entry, &ucs_plugin_registry, list) {
        if (strcmp(entry->component, component) == 0) {
            return entry;
        }
    }

    return NULL;
}

static ucs_status_t ucs_plugin_try_load(const char *plugin_name_or_path,
                                        void **handle)
{
    void *dl_handle;

    (void)dlerror();
    dl_handle = dlopen(plugin_name_or_path, RTLD_GLOBAL | RTLD_LAZY);
    if (dl_handle != NULL) {
        *handle = dl_handle;
        ucs_debug("Loaded plugin: %s", plugin_name_or_path);
        return UCS_OK;
    }

    return UCS_ERR_NO_DEVICE;
}

ucs_plugin_desc_t* ucs_plugin_load_component(const char *component)
{
    char plugin_name[128] = {0};
    ucs_plugin_registry_entry_t *entry;
    ucs_status_t status;
    void *handle;

    if (component == NULL) {
        return NULL;
    }

    UCS_INIT_ONCE(&ucs_plugin_registry_init_once) {
        /* first use - list already initialized statically */
    }

    entry = ucs_plugin_registry_find(component);
    if (entry != NULL) {
        return &entry->plugin;
    }

    snprintf(plugin_name, sizeof(plugin_name), "libucx_plugin_%s.so", component);
    status = ucs_plugin_try_load(plugin_name, &handle);
    if (status != UCS_OK) {
        ucs_debug("No plugin found for component '%s'", component);
        return NULL;
    }

    entry = ucs_malloc(sizeof(*entry), "plugin_registry_entry");
    if (entry == NULL) {
        goto err_close;
    }

    entry->component = ucs_strdup(component, "plugin_component");
    if (entry->component == NULL) {
        goto err_free_entry;
    }

    entry->plugin.component = entry->component;
    entry->plugin.handle    = handle;

    ucs_list_add_tail(&ucs_plugin_registry, &entry->list);

    return &entry->plugin;

err_free_entry:
    ucs_free(entry);
err_close:
    dlclose(handle);
    return NULL;
}

void ucs_plugin_free_descriptors(void)
{
    ucs_plugin_registry_entry_t *entry, *tmp;

    UCS_INIT_ONCE(&ucs_plugin_registry_init_once) {
        return; /* registry never initialized, nothing to clean */
    }

    ucs_list_for_each_safe(entry, tmp, &ucs_plugin_registry, list) {
        if (entry->plugin.handle != NULL) {
            dlclose(entry->plugin.handle);
        }

        ucs_free(entry->component);
        ucs_list_del(&entry->list);
        ucs_free(entry);
    }
}

#else /* UCX_SHARED_LIB */

/* Stub implementations for static builds */
ucs_plugin_desc_t* ucs_plugin_load_component(const char *component)
{
    return NULL;
}

void ucs_plugin_free_descriptors(void)
{
}

#endif /* UCX_SHARED_LIB */
