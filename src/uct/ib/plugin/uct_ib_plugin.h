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
 * Plugin capability flags (bitmap)
 */
typedef enum {
    UCT_IB_PLUGIN_CAP_NONE          = 0,
    UCT_IB_PLUGIN_CAP_HW_PSN        = UCS_BIT(0),  /**< Hardware PSN support */
    UCT_IB_PLUGIN_CAP_BATCH_SEND    = UCS_BIT(1), /**< Batch send support */
    UCT_IB_PLUGIN_CAP_HELLO_WORLD   = UCS_BIT(2), /**< Hello World feature */
} uct_ib_plugin_cap_flags_t;

/**
 * Plugin information structure
 */
typedef struct {
    const char *name;        /**< Plugin name */
    const char *version;     /**< Plugin version */
    const char *description; /**< Plugin description */
} uct_ib_plugin_info_t;

/**
 * Initialize the plugin
 *
 * @return UCS_OK on success, UCS_ERR_UNSUPPORTED if plugin is not available
 */
ucs_status_t ucx_plugin_init(void);

/**
 * Query plugin capabilities
 *
 * @param [out] capability_flags  Output parameter filled with bitmap of plugin capabilities
 * @return UCS_OK on success, UCS_ERR_UNSUPPORTED if plugin is not available
 */
ucs_status_t ucx_plugin_query(uint64_t *capability_flags);

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

/**
 * Hello World demonstration function
 *
 * @return UCS_OK on success, UCS_ERR_UNSUPPORTED if plugin is not available
 */
ucs_status_t ucx_plugin_hello(void);

#ifdef __cplusplus
}
#endif

#endif /* UCT_IB_PLUGIN_H_ */
