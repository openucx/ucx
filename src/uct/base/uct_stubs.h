/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_STUBS_H_
#define UCT_STUBS_H_

#include <uct/api/uct_def.h>

/* Stub definitions for UCT API */

#define uct_pending_purge_callback_assert \
        (uct_pending_purge_callback_t)ucs_empty_function_do_assert_void

#define uct_iface_event_fd_get_func_unsupported \
        (uct_iface_event_fd_get_func_t)ucs_empty_function_return_unsupported

#define uct_iface_event_arm_func_empty \
        (uct_iface_event_arm_func_t)ucs_empty_function_return_success

#define uct_iface_progress_enable_func_empty \
        (uct_iface_progress_enable_func_t)ucs_empty_function

#define uct_iface_progress_disable_func_empty \
        (uct_iface_progress_enable_func_t)ucs_empty_function

#define uct_iface_close_func_empty \
        (uct_iface_close_func_t)ucs_empty_function

#define uct_iface_get_address_func_empty \
        (uct_iface_get_address_func_t)ucs_empty_function_return_success

#define uct_ep_pending_purge_func_empty \
        (uct_ep_pending_purge_func_t)ucs_empty_function_return_success

#define uct_ep_pending_add_func_busy \
        (uct_ep_pending_add_func_t)ucs_empty_function_return_busy

#define uct_ep_put_short_func_unsupported \
        (uct_ep_put_short_func_t)ucs_empty_function_return_unsupported

#define uct_ep_get_bcopy_func_unsupported \
        (uct_ep_get_bcopy_func_t)ucs_empty_function_return_unsupported

#define uct_ep_atomic_cswap64_func_unsupported \
        (uct_ep_atomic_cswap64_func_t)ucs_empty_function_return_unsupported

#define uct_ep_atomic_cswap32_func_unsupported \
        (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_unsupported

#define uct_ep_atomic64_post_func_unsupported \
        (uct_ep_atomic64_post_func_t)ucs_empty_function_return_unsupported

#define uct_ep_atomic32_post_func_unsupported \
        (uct_ep_atomic32_post_func_t)ucs_empty_function_return_unsupported

#define uct_ep_atomic32_fetch_func_unsupported \
        (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_unsupported

#define uct_ep_atomic64_fetch_func_unsupported \
        (uct_ep_atomic64_fetch_func_t)ucs_empty_function_return_unsupported

#define uct_ep_check_func_unsupported \
        (uct_ep_check_func_t)ucs_empty_function_return_unsupported

#endif /* UCT_STUBS_H_ */
