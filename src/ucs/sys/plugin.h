/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PLUGIN_H_
#define UCS_PLUGIN_H_

#include <ucs/type/status.h>
#include <ucs/type/init_once.h>
#include <ucs/datastruct/array.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Plugin descriptor - metadata about a dynamically loaded plugin
 * 
 * Plugins are separate shared libraries that are built independently from UCX
 * and loaded at runtime. This structure holds metadata about a successfully
 * loaded plugin.
 */
typedef struct ucs_plugin_desc {
    const char *name;           /* Plugin name */
    const char *component;      /* Component name (e.g., "ib", "cuda") */
    void *handle;               /* dlopen() handle for the plugin library */
    void *data;                 /* Component-specific plugin data */
} ucs_plugin_desc_t;

/**
 * Plugin array type - array of plugin descriptors
 */
UCS_ARRAY_DECLARE_TYPE(ucs_plugin_array_t, unsigned, ucs_plugin_desc_t);

/**
 * Plugin loader configuration
 * 
 * Used to configure how plugins are discovered and loaded for a specific component.
 * Plugins are built separately and installed to standard library paths, then discovered
 * and loaded dynamically at runtime by UCX.
 * 
 * Note: We don't include flags or max_plugins because:
 * - flags: Always use RTLD_GLOBAL | RTLD_LAZY (needed for weak symbol override)
 * - max_plugins: No limit needed - allocate memory as plugins are discovered
 */
typedef struct ucs_plugin_loader_config {
    const char *component;      /* Component name to load plugins for (e.g., "ib") */
    const char *plugin_prefix;  /* Plugin library prefix (default: "libucx_plugin") */
} ucs_plugin_loader_config_t;

/**
 * Load plugins for a component
 * 
 * This function searches for and dynamically loads plugins matching the specified pattern.
 * Plugins are built separately from UCX and must be installed to standard library paths.
 * 
 * Plugins are always loaded with RTLD_GLOBAL | RTLD_LAZY to ensure:
 * - Plugin symbols override weak stubs (RTLD_GLOBAL)
 * - Symbol resolution is deferred for better startup performance (RTLD_LAZY)
 * 
 * There is no limit on the number of plugins - all discovered plugins are loaded.
 * 
 * Search order:
 * 1. Standard library paths (LD_LIBRARY_PATH, system paths like /usr/lib, /usr/local/lib)
 * 2. UCX installation directory ($prefix/lib/ucx/)
 * 3. Relative to component library (same directory as component .so)
 * 
 * Naming conventions:
 * - Standard: libucx_plugin_<name>.so (e.g., libucx_plugin_ib.so)
 * 
 * @param [in]  config      Plugin loader configuration
 * 
 * @return Pointer to plugin array if successful (even if empty), NULL on error
 * @note The returned array is managed internally and should not be freed by the caller
 */
ucs_plugin_array_t* ucs_plugin_load_component(const ucs_plugin_loader_config_t *config);

/**
 * Find a specific plugin by name
 * 
 * @param [in] component Component name
 * @param [in] name      Plugin name
 * 
 * @return Plugin descriptor if found, NULL otherwise
 */
ucs_plugin_desc_t* ucs_plugin_find(const char *component, const char *name);

/**
 * Free plugin descriptors and unload plugin libraries
 * 
 * This function cleans up the entire plugin registry, including:
 * - Closing all dlopen() handles for loaded plugins
 * - Freeing all allocated plugin name strings
 * - Freeing all component strings
 * - Freeing all plugin arrays
 * - Freeing all registry entries
 * 
 * @note This function should only be called when plugins are no longer needed.
 *       In most cases, plugins remain loaded for the lifetime of the process.
 */
void ucs_plugin_free_descriptors(void);

#ifdef __cplusplus
}
#endif

#endif /* UCS_PLUGIN_H_ */
