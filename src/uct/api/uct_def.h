/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_DEF_H_
#define UCT_DEF_H_

#include <ucs/sys/math.h>
#include <ucs/type/status.h>
#include <stdint.h>


#define UCT_TL_NAME_MAX          8
#define UCT_PD_COMPONENT_NAME_MAX  8
#define UCT_PD_NAME_MAX          16
#define UCT_DEVICE_NAME_MAX      32
#define UCT_AM_ID_BITS           5
#define UCT_AM_ID_MAX            UCS_BIT(UCT_AM_ID_BITS)
#define UCT_INVALID_MEM_HANDLE   NULL
#define UCT_INVALID_RKEY         ((uintptr_t)(-1))
#define UCT_INLINE_API           static UCS_F_ALWAYS_INLINE


typedef struct uct_iface         *uct_iface_h;
typedef struct uct_iface_config  uct_iface_config_t;
typedef struct uct_ep            *uct_ep_h;
typedef void *                   uct_mem_h;
typedef uintptr_t                uct_rkey_t;
typedef struct uct_pd            *uct_pd_h;
typedef struct uct_pd_ops        uct_pd_ops_t;
typedef void                     *uct_rkey_ctx_h;
typedef struct uct_iface_attr    uct_iface_attr_t;
typedef struct uct_pd_attr       uct_pd_attr_t;
typedef struct uct_completion    uct_completion_t;
typedef struct uct_worker        *uct_worker_h;
typedef struct uct_pd            uct_pd_t;


/**
 * Callback to process incoming active message
 *
 * When the callback is called, `desc' does not necessarily contain the payload.
 * In this case, `data' would not point inside `desc', and user may want copy the
 * payload from `data' to `desc' before returning UCT_INPROGRESS (it's guaranteed
 * `desc' has enough room to hold the payload).
 *
 * @param [in]  arg      User-defined argument.
 * @param [in]  data     Points to the received data.
 * @param [in]  length   Length of data.
 * @param [in]  desc     Points to the received descriptor, at the beginning of
 *                       the user-defined rx_headroom.
 *
 * @return UCS_OK - descriptor was consumed, and can be released by the caller.
 *         UCS_INPROGRESS - descriptor is owned by the callee, and would be released later.
 */
typedef ucs_status_t (*uct_am_callback_t)(void *arg, void *data, size_t length,
                                          void *desc);


/**
 * Callback to process send completion.
 *
 * @param [in]  self     Pointer to relevant completion structure, which was
 *                       initially passed to the operation.
 */
typedef void (*uct_completion_callback_t)(uct_completion_t *self);


/**
 * Callback for producing data.
 *
 * @param [in]  dest     Memory buffer to pack the data to.
 * @param [in]  arg      Custom user-argument.
 * @param [in]  length   How much data to produce (size of "dest")
 *
 * @note The arguments for this callback are in the same order as libc's memcpy().
 */
typedef void (*uct_pack_callback_t)(void *dest, void *arg, size_t length);


/**
 * Callback for consuming data.
 *
 * @param [in]  arg      Custom user-argument.
 * @param [in]  data     Memory buffer to unpack the data from.
 * @param [in]  length   How much data to consume (size of "data")
 *
 * @note The arguments for this callback are in the same order as libc's memcpy().
 */
typedef void (*uct_unpack_callback_t)(void *arg, const void *data, size_t length);


#endif
