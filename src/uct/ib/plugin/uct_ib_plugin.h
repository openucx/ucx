/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_PLUGIN_H_
#define UCT_IB_PLUGIN_H_

#include <ucs/type/status.h>
#include <uct/api/uct.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Plugin information structure
 */
typedef struct {
    const char *name;        /**< Plugin name */
    const char *description; /**< Plugin description */
    const char *ucx_commit;  /**< UCX commit hash */
    uint64_t version_major;  /**< Plugin version major */
    uint64_t version_minor;  /**< Plugin version minor */
    uint64_t version_patch;  /**< Plugin version patch */
} uct_ib_plugin_info_t;

/**
 * Initialize the plugin
 *
 * @return UCS_OK on success, UCS_ERR_UNSUPPORTED if plugin is not available
 */
ucs_status_t ucx_plugin_init(void);

/**
 * Cleanup/teardown the plugin
 */
void ucx_plugin_cleanup(void);

/**
 * Get plugin information
 *
 * @return Pointer to plugin information structure, or NULL if plugin is not available
 */
const uct_ib_plugin_info_t* ucx_plugin_get_info(void);

#ifdef __cplusplus
}
#endif

#endif /* UCT_IB_PLUGIN_H_ */
