/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_H_
#define UCT_H_


#include "tl.h"


/**
 * @ingroup CONTEXT
 * @brief Initialize global context.
 *
 * @param [out] context_p   Filled with context handle.
 *
 * @return Error code.
 */
ucs_status_t uct_init(uct_context_h *context_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy global context.
 *
 * @param [in] context   Handle to context.
 */
void uct_cleanup(uct_context_h context);


/**
 * @ingroup CONTEXT
 * @brief Query for transport resources.
 *
 * @param [in]  context         Handle to context.
 * @param [out] resources_p     Filled with a pointer to an array of resource descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_query_resources(uct_context_h context,
                                 uct_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);


/**
 * @ingroup CONTEXT
 * @brief Release the list of resources returned from uct_query_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 *
 */
void uct_release_resource_list(uct_resource_desc_t *resources);


/**
 * @brief Open a communication interface.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  tl_name       Transport name.
 * @param [in]  hw_name       Hardware resource name,
 * @param [out] iface_p       Filled with a handle to opened communication interface.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_open(uct_context_h context, const char *tl_name,
                            const char *hw_name, uct_iface_h *iface_p);

void uct_iface_close(uct_iface_h iface);

#endif
