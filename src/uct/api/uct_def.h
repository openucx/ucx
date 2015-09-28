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


#define UCT_TL_NAME_MAX          10
#define UCT_PD_COMPONENT_NAME_MAX  8
#define UCT_PD_NAME_MAX          16
#define UCT_DEVICE_NAME_MAX      32
#define UCT_PENDING_REQ_PRIV_LEN 16
#define UCT_AM_ID_BITS           5
#define UCT_AM_ID_MAX            UCS_BIT(UCT_AM_ID_BITS)
#define UCT_INVALID_MEM_HANDLE   NULL
#define UCT_INVALID_RKEY         ((uintptr_t)(-1))
#define UCT_INLINE_API           static UCS_F_ALWAYS_INLINE


/**
 * @ingroup AM
 * @brief Trace types for active message tracer.
 */
enum uct_am_trace_type {
    UCT_AM_TRACE_TYPE_SEND,
    UCT_AM_TRACE_TYPE_RECV,
    UCT_AM_TRACE_TYPE_LAST
};


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
typedef struct uct_pending_req  uct_pending_req_t;
typedef struct uct_worker        *uct_worker_h;
typedef struct uct_pd            uct_pd_t;
typedef enum uct_am_trace_type   uct_am_trace_type_t;


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
 * Callback to trace active messages. Writes a string which represents active
 * message contents into 'buffer'.
 *
 * @param [in]  arg      User-defined argument.
 * @param [in]  type     Message type.
 * @param [in]  id       Active message id.
 * @param [in]  data     Points to the received data.
 * @param [in]  length   Length of data.
 * @param [out] buffer   Filled with a debug information string.
 * @param [in]  max      Maximal length of the string.
 */
typedef void (*uct_am_tracer_t)(void *arg, uct_am_trace_type_t type, uint8_t id,
                                const void *data, size_t length, char *buffer,
                                size_t max);


/**
 * Callback to process send completion.
 *
 * @param [in]  self     Pointer to relevant completion structure, which was
 *                       initially passed to the operation.
 */
typedef void (*uct_completion_callback_t)(uct_completion_t *self);


/**
 * Callback to process pending requests.
 *
 * @param [in]  self     Pointer to relevant pending structure, which was
 *                       initially passed to the operation.
 *
 * @return UCS_OK              - This pending request should be removed.
 *         UCS_INPROGRESS      - Keep this pending request on the queue, and
 *                               continue to next requests.
 *         < 0 (UCS_ERR_xx)    - Keep this pending request on the queue, and
 *                               stop processing the queue.
 */
typedef ucs_status_t (*uct_pending_callback_t)(uct_pending_req_t *self);


/**
 * Callback for producing data.
 *
 * @param [in]  dest     Memory buffer to pack the data to.
 * @param [in]  arg      Custom user-argument.
 *
 * @return  How much data was actually produced.
 */
typedef size_t (*uct_pack_callback_t)(void *dest, void *arg);


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
