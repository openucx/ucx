/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_DEF_H_
#define UCT_DEF_H_

#include <ucs/sys/math.h>
#include <ucs/type/status.h>
#include <stdint.h>


#define UCT_MAX_NAME_LEN         64
#define UCT_AM_ID_BITS           5
#define UCT_AM_ID_MAX            UCS_BIT(UCT_AM_ID_BITS)
#define UCT_INVALID_MEM_KEY      ((uintptr_t)0)
#define UCT_INLINE_API           static UCS_F_ALWAYS_INLINE


typedef struct uct_context       *uct_context_h;
typedef struct uct_iface         *uct_iface_h;
typedef struct uct_iface_addr    uct_iface_addr_t;
typedef struct uct_iface_config  uct_iface_config_t;
typedef struct uct_ep            *uct_ep_h;
typedef struct uct_ep_addr       uct_ep_addr_t;
typedef uintptr_t                uct_lkey_t;
typedef uintptr_t                uct_rkey_t;
typedef struct uct_pd            *uct_pd_h;
typedef struct uct_tl_ops        uct_tl_ops_t;
typedef struct uct_pd_ops        uct_pd_ops_t;
typedef void                     *uct_rkey_ctx_h;
typedef struct uct_iface_attr    uct_iface_attr_t;
typedef struct uct_pd_attr       uct_pd_attr_t;
typedef struct uct_completion    uct_completion_t;


/**
 * Remote key release function.
 */
typedef void (*uct_rkey_release_func_t)(uct_context_h context, uct_rkey_t rkey);


/**
 * Callback to process incoming data buffer.
 * Used for atomics, get bcopy.
 *
 * @param [in]  desc     Points to the received descriptor, just after rx_headroom.
 * @param [in]  data     Points to the received data.
 * @param [in]  length   Length of data.
 * @param [in]  arg      User-defined argument.
 *
 * @return UCS_OK - descriptor was consumed, and can be released by the caller.
 *         UCS_INPROGRESS - descriptor is owned by the callee, and would be released later.
 */
typedef ucs_status_t (*uct_bcopy_recv_callback_t)(void *desc, void *data,
                                                  size_t length, void *arg);


/**
 * Callback to process incoming immediate data.
 * Used for atomics.
 *
 * @param [in]  arg      User-defined argument.
 * @param [in]  data     Data received from remote.
 */
typedef void (*uct_imm_recv_callback_t)(void *arg, uint64_t data);


/**
 * Callback for producing data.
 *
 * @param [in]  dest     Memory buffer to pack the data to.
 * @param [in]  arg      Custom user-argument.
 * @param [in]  length   How much data to produce (size of "dest")
 *
 * @note The arguments for this callback are in the same order as libc's memcpy().
 *
 * TODO return How much data was actually produced.
 */
typedef void (*uct_pack_callback_t)(void *dest, void *arg, size_t length);


#endif
