/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_COMPAT_H_
#define UCP_COMPAT_H_


#include <ucp/api/ucp_def.h>
#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS


/**
 * @ingroup UCP_WORKER
 * @deprecated Replaced by @ref ucp_listener_accept_conn_handler_t.
 */
typedef struct ucp_listener_accept_handler {
   ucp_listener_accept_callback_t  cb;       /**< Endpoint creation callback */
   void                            *arg;     /**< User defined argument for the
                                                  callback */
} ucp_listener_accept_handler_t;


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_request_test.
 */
int ucp_request_is_completed(void *request);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_request_free.
 */
void ucp_request_release(void *request);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_ep_close_nb.
 */
void ucp_ep_destroy(ucp_ep_h ep);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_ep_close_nb.
 */
ucs_status_ptr_t ucp_disconnect_nb(ucp_ep_h ep);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_tag_recv_request_test and
 *             @ref ucp_request_check_status depends on use case.
 *
 * @note Please use @ref ucp_request_check_status for cases that only need to
 *       check the completion status of an outstanding request.
 *       @ref ucp_request_check_status can be used for any type of request.
 *       @ref ucp_tag_recv_request_test should only be used for requests
 *       returned by @ref ucp_tag_recv_nb (or request allocated by user for
 *       @ref ucp_tag_recv_nbr) for which additional information
 *       (returned via the @a info pointer) is needed.
 */
ucs_status_t ucp_request_test(void *request, ucp_tag_recv_info_t *info);


/**
 * @ingroup UCP_ENDPOINT
 * @deprecated Replaced by @ref ucp_ep_flush_nb.
 */
ucs_status_t ucp_ep_flush(ucp_ep_h ep);

/**
 * @ingroup UCP_WORKER
 *
 * @brief Flush outstanding AMO and RMA operations on the @ref ucp_worker_h
 * "worker"
 *
 * This routine flushes all outstanding AMO and RMA communications on the
 * @ref ucp_worker_h "worker". All the AMO and RMA operations issued on the
 * @a worker prior to this call are completed both at the origin and at the
 * target when this call returns.
 *
 * @note For description of the differences between @ref ucp_worker_flush
 * "flush" and @ref ucp_worker_fence "fence" operations please see
 * @ref ucp_worker_fence "ucp_worker_fence()"
 *
 * @param [in] worker        UCP worker.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucp_worker_flush(ucp_worker_h worker);


/**
 * @ingroup UCP_ENDPOINT
 * @brief Modify endpoint parameters.
 *
 * This routine modifies @ref ucp_ep_h "endpoint" created by @ref ucp_ep_create
 * or @ref ucp_listener_accept_callback_t. For example, this API can be used
 * to setup custom parameters like @ref ucp_ep_params_t::user_data or
 * @ref ucp_ep_params_t::err_handler_cb to endpoint created by 
 * @ref ucp_listener_accept_callback_t.
 *
 * @param [in]  ep          A handle to the endpoint.
 * @param [in]  params      User defined @ref ucp_ep_params_t configurations
 *                          for the @ref ucp_ep_h "UCP endpoint".
 *
 * @return NULL             - The endpoint is modified successfully.
 * @return UCS_PTR_IS_ERR(_ptr) - The reconfiguration failed and an error code
 *                                indicates the status. However, the @a endpoint
 *                                is not modified and can be used further.
 * @return otherwise        - The reconfiguration process is started, and can be
 *                            completed at any point in time. A request handle
 *                            is returned to the application in order to track
 *                            progress of the endpoint modification.
 *                            The application is responsible for releasing the
 *                            handle using the @ref ucp_request_free routine.
 *
 * @note See the documentation of @ref ucp_ep_params_t for details, only some of
 *       the parameters can be modified.
 * @deprecated Use @ref ucp_listener_accept_conn_handler_t instead of @ref
 *             ucp_listener_accept_handler_t, if you have other use case please
 *             submit an issue on https://github.com/openucx/ucx or report to
 *             ucx-group@elist.ornl.gov
 */
ucs_status_ptr_t ucp_ep_modify_nb(ucp_ep_h ep, const ucp_ep_params_t *params);


END_C_DECLS

#endif
