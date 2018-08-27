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
 * @deprecated Replaced by @ref ucp_worker_flush_nb.
 */
ucs_status_t ucp_worker_flush(ucp_worker_h worker);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_put_nb.
 */
ucs_status_t ucp_put(ucp_ep_h ep, const void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_get_nb.
 */
ucs_status_t ucp_get(ucp_ep_h ep, void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_post.
 */
ucs_status_t ucp_atomic_add32(ucp_ep_h ep, uint32_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_post.
 */
ucs_status_t ucp_atomic_add64(ucp_ep_h ep, uint64_t add,
                              uint64_t remote_addr, ucp_rkey_h rkey);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb.
 */
ucs_status_t ucp_atomic_fadd32(ucp_ep_h ep, uint32_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb.
 */
ucs_status_t ucp_atomic_fadd64(ucp_ep_h ep, uint64_t add, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb.
 */
ucs_status_t ucp_atomic_swap32(ucp_ep_h ep, uint32_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb.
 */
ucs_status_t ucp_atomic_swap64(ucp_ep_h ep, uint64_t swap, uint64_t remote_addr,
                               ucp_rkey_h rkey, uint64_t *result);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb.
 */
ucs_status_t ucp_atomic_cswap32(ucp_ep_h ep, uint32_t compare, uint32_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey,
                                uint32_t *result);


/**
 * @ingroup UCP_COMM
 * @deprecated Replaced by @ref ucp_atomic_fetch_nb.
 */
ucs_status_t ucp_atomic_cswap64(ucp_ep_h ep, uint64_t compare, uint64_t swap,
                                uint64_t remote_addr, ucp_rkey_h rkey,
                                uint64_t *result);

END_C_DECLS

#endif
