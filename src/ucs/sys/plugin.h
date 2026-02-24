/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PLUGIN_H_
#define UCS_PLUGIN_H_

#include <ucs/type/status.h>
#include <ucs/type/init_once.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Plugin descriptor - metadata about a loaded plugin */
typedef struct ucs_plugin_desc {
    const char *component;      /* Component name (e.g., "ib") */
    void *handle;               /* dlopen() handle */
} ucs_plugin_desc_t;

/**
 * Load plugin for a component by searching for libucx_plugin_<component>.so.
 * Returns cached descriptor if already loaded. Uses RTLD_GLOBAL | RTLD_LAZY.
 *
 * @param [in] component  Component name (e.g., "ib")
 * @return Plugin descriptor, or NULL if not found
 */
ucs_plugin_desc_t* ucs_plugin_load_component(const char *component);

/**
 * Free all plugin descriptors and close dlopen handles.
 */
void ucs_plugin_free_descriptors(void);

#ifdef __cplusplus
}
#endif

#endif /* UCS_PLUGIN_H_ */
