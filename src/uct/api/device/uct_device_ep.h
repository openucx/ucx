#ifndef UCT_DEVICE_EP_H
#define UCT_DEVICE_EP_H

#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <stdint.h>
#include <stdlib.h>

#include "uct_device_types.h"

/**
 * @defgroup UCT_DEVICE Device API
 * @ingroup UCT_API
 * @{
 * * This section describes UCT Device API.
 * @}
 */

/**
 * @ingroup UCT_DEVICE
 * @brief Posts one memory put operation.
 *
 * This GPU routine posts single buffer using put operation.
 * The addresses and length must be valid for the used @a mem_elem.
 *
 * User can pass @a comp to track execution and resources.
 * The flags parameter can be used to modify the behavior
 * of the routine.
 *
 * @param [in]  device_ep       Exported ep to be used for the operation.
 * @param [in]  mem_elem        Memory context for transaction.
 * @param [in]  address         Local virtual address to send data from.
 * @param [in]  remote_address  Remote virtual address to send data to.
 * @param [in]  length          Length in bytes of the data to send.
 * @param [in]  flags           Flags usable to modify the function behavior.
 * @param [in]  comp            Object to track resources and execution of operation.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
UCS_F_DEVICE ucs_status_t uct_device_ep_put_single(
        uct_device_ep_h device_ep, const uct_mem_element_h mem_elem,
        const void *address, uint64_t *remote_address, size_t length,
        uint64_t flags, uct_dev_completion_t *comp)
{
    // TODO - add switch statetment and call ep specific put function
    return UCS_OK;
}

#endif
