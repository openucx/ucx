/**
* Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_GAUDI_TOPO_H
#define UCT_GAUDI_TOPO_H

#include <ucs/sys/topo/base/topo.h>
#include <ucs/type/status.h>

/**
 * @defgroup UCT_GAUDI_INTERNAL Internal Gaudi Topology Helpers
 * @brief UNSTABLE – Do not include from external code
 *
 * This header defines an *unstable* internal interface for the Gaudi transport.
 * It is NOT installed and may change or be removed without notice.
 *
 * External middleware (e.g., NIXL) should forward-declare any needed prototypes
 * rather than including this file. Only the implementation inside UCX may use it.
 *
 * @note Requires configure --enable-gaudi-topo-api to build the provider.
 */

BEGIN_C_DECLS

/**
 * Initialize Gaudi topology provider.
 *
 * @note Must be paired with uct_gaudi_topo_cleanup().
 */
void uct_gaudi_topo_init(void);

/**
 * Clean up Gaudi topology provider.
 *
 * @note Safe to call if topology is not initialized.
 */
void uct_gaudi_topo_cleanup(void);

/**
 * Get Gaudi device index from a given module ID.
 *
 * Searches /sys/class/accel for the "accel<N>" directory whose module_id
 * attribute matches the supplied value and returns the numeric index <N>.
 *
 * @param [in] module_id  Gaudi module identifier to query.
 *
 * @return Non-negative Device index on success, -1 on failure
 *         (error details printed via ucs_error).
 *
 * @note On success, the return value is a zero-based index parsed from the
 *       "accel<N>" directory name. On failure, -1 is returned and the error
 *       is logged using ucs_error().
 */
int uct_gaudi_get_index_from_module_id(uint32_t module_id);

/**
 * Find best HNIC for a Gaudi device based on topology distance.
 *
 * @param [in]  accel_name  Name of the Gaudi device (e.g., "accel0").
 * @param [out] hnic_device Filled with selected HNIC device ID.
 * @param [out] port_num    Filled with the default UCX port for that NIC.
 *
 * @return UCS_OK on success,
 * @return UCS_ERR_INVALID_PARAM if any parameter is NULL,
 * @return UCS_ERR_NO_ELEM if no suitable HNIC is found,
 * @return other UCX error codes on initialization failures.
 */
ucs_status_t uct_gaudi_find_best_connection(const char *accel_name,
                                            ucs_sys_device_t *hnic_device,
                                            int *port_num);

END_C_DECLS

#endif /* UCT_GAUDI_TOPO_H */
