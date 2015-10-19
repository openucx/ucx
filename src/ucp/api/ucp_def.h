/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCP_DEF_H_
#define UCP_DEF_H_

#include <ucs/type/status.h>
#include <stddef.h>
#include <stdint.h>

#define UCP_REQUEST_PRIV_LEN             80


typedef struct ucp_params                ucp_params_t;
typedef struct ucp_context               *ucp_context_h;
typedef struct ucp_ep                    *ucp_ep_h;
typedef void                             ucp_address_t;
typedef struct ucp_tag_recv_info         ucp_tag_recv_info_t;
typedef struct ucp_generic_dt_ops        ucp_generic_dt_ops_t;
typedef struct ucp_rkey                  *ucp_rkey_h;
typedef struct ucp_mem                   *ucp_mem_h;
typedef struct ucp_worker                *ucp_worker_h;
typedef uint64_t                         ucp_tag_t;
typedef uint64_t                         ucp_datatype_t;


/**
 * @ingroup UCP_WORKER
 * @brief Progress callback. Used to progress user context during blocking operations.
 *
 * @param [in]  arg       User-defined argument.
 */
typedef void (*ucp_user_progress_func_t)(void *arg);


/**
 * @ingroup UCP_CONTEXT
 * Callback to initialize the use request structure.
 *
 * @param [in]  request   Request handle to initialize.
 */
typedef void (*ucp_request_init_callback_t)(void *request);


/**
 * @ingroup UCP_CONTEXT
 * Callback to cleanup a request structure (called just before memory is
 *  finally released, not every time a request is released).
 *
 * @param [in]  request   Request handle to cleanup.
 */
typedef void (*ucp_request_cleanup_callback_t)(void *request);


/**
 * @ingroup UCP_COMM
 * @brief Completion callback for non-blocking sends.
 *
 * @param [in]  request   The completed send request.
 * @param [in]  status    Completion status:
 *                           UCS_OK           - completed successfully.
 *                           UCS_ERR_CANCELED - send was canceled.
 *                           otherwise        - error during send.
 */
typedef void (*ucp_send_callback_t)(void *request, ucs_status_t status);


/**
 * @ingroup UCP_COMM
 * @brief Completion callback for non-blocking tag receives.
 *
 * @param [in]  request   The completed receive request.
 * @param [in]  status    Completion status.
 *                           UCS_OK             - completed successfully.
 *                           UCS_ERR_TRUNCATRED - data could not fit to buffer.
 *                           UCS_ERR_CANCELED   - receive was canceled.
 *                           otherwise          - error during receive.
 * @param [in]  info      Completion information (tag, length). Valid only id
 *                        status is UCS_OK.
 */
typedef void (*ucp_tag_recv_callback_t)(void *request, ucs_status_t status,
                                        ucp_tag_recv_info_t *info);


#endif
