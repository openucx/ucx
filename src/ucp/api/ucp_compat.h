/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_COMPAT_H_
#define UCP_COMPAT_H_


#include <ucp/api/ucp_def.h>


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
 * @note Please, use @ref ucp_request_check_status in cases then only check
 *       status is needed for any type of request and
 *       @ref ucp_tag_recv_request_test for request returned from
 *       @ref ucp_tag_recv_nb routine and out-parameter @a info is required.
 */
ucs_status_t ucp_request_test(void *request, ucp_tag_recv_info_t *info);


#endif
