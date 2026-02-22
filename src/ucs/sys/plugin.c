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
#include <ucs/datastruct/array.h>
#include <ucs/datastruct/list.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <dlfcn.h>

#ifdef UCX_SHARED_LIB

/* Plugin registry - stores loaded plugins per component */
typedef struct ucs_plugin_registry_entry {
    ucs_list_link_t list;
    char *component;
    ucs_plugin_array_t plugins;  /* Array of plugins for this component */
} ucs_plugin_registry_entry_t;

static ucs_init_once_t ucs_plugin_registry_init_once = UCS_INIT_ONCE_INITIALIZER;
static UCS_LIST_HEAD(ucs_plugin_registry);

/**
 * Find or create registry entry for component
 * Note: This is called from within UCS_INIT_ONCE, so it's thread-safe
 */
static ucs_plugin_registry_entry_t* ucs_plugin_registry_get(const char *component)
{
    ucs_plugin_registry_entry_t *entry;

    /* Find existing entry */
    ucs_list_for_each(entry, &ucs_plugin_registry, list) {
        if (strcmp(entry->component, component) == 0) {
            return entry;
        }
    }

    /* Create new entry */
    entry = ucs_malloc(sizeof(*entry), "plugin_registry_entry");
    if (entry == NULL) {
        return NULL;
    }

    entry->component = ucs_strdup(component, "plugin_component");
    if (entry->component == NULL) {
        ucs_free(entry);
        return NULL;
    }

    ucs_array_init_dynamic(&entry->plugins);
    ucs_list_add_tail(&ucs_plugin_registry, &entry->list);

    return entry;
}

/**
 * Try to load a plugin library by name
 * 
 * @param [in]  plugin_name Plugin library name (e.g., "libucx_plugin_ib.so")
 * @param [out] handle      dlopen() handle if successful
 * 
 * @return UCS_OK if loaded, UCS_ERR_NO_DEVICE if not found, other on error
 */
/**
 * Try to load a plugin library
 * 
 * Can load by name (searches standard library paths) or by full path.
 * dlopen() handles both cases automatically.
 * 
 * @param [in]  plugin_name_or_path Plugin library name (e.g., "libucx_plugin_ib.so")
 *                                  or full path to plugin library
 * @param [out] handle              dlopen() handle if successful
 * 
 * @return UCS_OK if loaded, UCS_ERR_NO_DEVICE if not found, other on error
 */
static ucs_status_t ucs_plugin_try_load(const char *plugin_name_or_path, void **handle)
{
    void *dl_handle;

    /* Clear any previous dlopen errors */
    (void)dlerror();

    /* dlopen() can handle both library names (searches standard paths) and full paths */
    dl_handle = dlopen(plugin_name_or_path, RTLD_GLOBAL | RTLD_LAZY);
    if (dl_handle != NULL) {
        *handle = dl_handle;
        ucs_debug("Loaded plugin: %s", plugin_name_or_path);
        return UCS_OK;
    }

    return UCS_ERR_NO_DEVICE;
}

/**
 * Load plugin by searching standard locations
 * Loads the first matching plugin found (standard naming or backward-compatible)
 * Adds to the end of the plugin array
 */
static ucs_status_t ucs_plugin_load_by_search(const char *component,
                                               const char *plugin_prefix,
                                               ucs_plugin_array_t *plugins_array)
{
    char plugin_name[128] = {0};  /* Plugin names are expected to be < 128 bytes */
    void *handle;
    ucs_status_t status;
    ucs_plugin_desc_t *plugin;

    /* Try standard naming: libucx_plugin_<component>.so */
    snprintf(plugin_name, sizeof(plugin_name), "%s_%s.so", plugin_prefix, component);

    /* Try loading from standard library paths */
    status = ucs_plugin_try_load(plugin_name, &handle);
    if (status != UCS_OK) {
        return UCS_ERR_NO_DEVICE;
    }

    /* Append new element to array */
    plugin = ucs_array_append(plugins_array, {
        return UCS_ERR_NO_MEMORY;
    });
    if (plugin == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    plugin->name = ucs_strdup(component, "plugin_name");
    plugin->component = component;
    plugin->handle = handle;
    plugin->data = NULL;

    if (plugin->name == NULL) {
        ucs_array_pop_back(plugins_array);
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

ucs_plugin_array_t* ucs_plugin_load_component(const ucs_plugin_loader_config_t *config)
{
    ucs_status_t status;
    ucs_plugin_registry_entry_t *entry;

    if (config == NULL) {
        return NULL;
    }

    if (config->component == NULL || config->plugin_prefix == NULL) {
        return NULL;
    }

    /* Initialize registry once */
    UCS_INIT_ONCE(&ucs_plugin_registry_init_once) {
        /* Registry initialized on first use */
    }

    /* Check if already loaded - get or create registry entry */
    entry = ucs_plugin_registry_get(config->component);
    if (entry == NULL) {
        return NULL;
    }

    /* Check if plugins already loaded */
    if (!ucs_array_is_empty(&entry->plugins)) {
        return &entry->plugins;
    }

    /* Try loading by searching standard locations */
    status = ucs_plugin_load_by_search(config->component, config->plugin_prefix,
                                       &entry->plugins);
    if (status == UCS_OK) {
        /* Update component pointers to point to registry entry's component string */
        ucs_plugin_desc_t *plugin;
        ucs_array_for_each(plugin, &entry->plugins) {
            plugin->component = entry->component;
        }
        return &entry->plugins;
    }

    /* No plugins found - this is OK, will use weak stubs */
    ucs_debug("No plugins found for component '%s'", config->component);
    return &entry->plugins;  /* Return empty array */
}

ucs_plugin_desc_t* ucs_plugin_find(const char *component, const char *name)
{
    ucs_plugin_registry_entry_t *entry;
    ucs_plugin_desc_t *plugin;

    if (component == NULL || name == NULL) {
        return NULL;
    }

    /* Initialize registry if needed */
    UCS_INIT_ONCE(&ucs_plugin_registry_init_once) {
        /* Registry initialized on first use */
    }

    entry = ucs_plugin_registry_get(component);
    if (entry == NULL || ucs_array_is_empty(&entry->plugins)) {
        return NULL;
    }

    ucs_array_for_each(plugin, &entry->plugins) {
        if (strcmp(plugin->name, name) == 0) {
            return plugin;
        }
    }

    return NULL;
}


#else /* UCX_SHARED_LIB */

/* Stub implementations for static builds */
ucs_plugin_array_t* ucs_plugin_load_component(const ucs_plugin_loader_config_t *config)
{
    return NULL;
}

ucs_plugin_desc_t* ucs_plugin_find(const char *component, const char *name)
{
    return NULL;
}

void ucs_plugin_free_descriptors(ucs_plugin_desc_t *plugins, unsigned num_plugins)
{
}

#endif /* UCX_SHARED_LIB */
