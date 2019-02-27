/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_DEF_H_
#define UCT_DEF_H_

#include <ucs/config/types.h>
#include <ucs/type/status.h>

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>


#define UCT_TL_NAME_MAX            10
#define UCT_MD_COMPONENT_NAME_MAX  16
#define UCT_MD_NAME_MAX            16
#define UCT_DEVICE_NAME_MAX        32
#define UCT_PENDING_REQ_PRIV_LEN   40
#define UCT_TAG_PRIV_LEN           32
#define UCT_AM_ID_BITS             5
#define UCT_AM_ID_MAX              UCS_BIT(UCT_AM_ID_BITS)
#define UCT_MEM_HANDLE_NULL        NULL
#define UCT_INVALID_RKEY           ((uintptr_t)(-1))
#define UCT_INLINE_API             static UCS_F_ALWAYS_INLINE


/**
 * @ingroup UCT_AM
 * @brief Trace types for active message tracer.
 */
enum uct_am_trace_type {
    UCT_AM_TRACE_TYPE_SEND,
    UCT_AM_TRACE_TYPE_RECV,
    UCT_AM_TRACE_TYPE_SEND_DROP,
    UCT_AM_TRACE_TYPE_RECV_DROP,
    UCT_AM_TRACE_TYPE_LAST
};


/**
 * @ingroup UCT_RESOURCE
 * @brief Flags for active message and tag-matching offload callbacks (callback's parameters).
 */
enum uct_cb_param_flags {

    /**
     * If this flag is enabled, then data is part of a descriptor which includes
     * the user-defined rx_headroom, and the callback may return UCS_INPROGRESS
     * and hold on to that descriptor. Otherwise, the data can't be used outside
     * the callback. If needed, the data must be copied-out.
     *
       @verbatim
       descriptor    data
       |             |
       +-------------+-------------------------+
       | rx_headroom | payload                 |
       +-------------+-------------------------+
       @endverbatim
     *
     */
    UCT_CB_PARAM_FLAG_DESC = UCS_BIT(0)
};

/**
 * @addtogroup UCT_RESOURCE
 * @{
 */
typedef struct uct_iface         *uct_iface_h;
typedef struct uct_iface_config  uct_iface_config_t;
typedef struct uct_md_config     uct_md_config_t;
typedef struct uct_ep            *uct_ep_h;
typedef void *                   uct_mem_h;
typedef uintptr_t                uct_rkey_t;
typedef struct uct_md            *uct_md_h;          /**< @brief Memory domain handler */
typedef struct uct_md_ops        uct_md_ops_t;
typedef void                     *uct_rkey_ctx_h;
typedef struct uct_iface_attr    uct_iface_attr_t;
typedef struct uct_iface_params  uct_iface_params_t;
typedef struct uct_md_attr       uct_md_attr_t;
typedef struct uct_completion    uct_completion_t;
typedef struct uct_pending_req   uct_pending_req_t;
typedef struct uct_worker        *uct_worker_h;
typedef struct uct_md            uct_md_t;
typedef enum uct_am_trace_type   uct_am_trace_type_t;
typedef struct uct_device_addr   uct_device_addr_t;
typedef struct uct_iface_addr    uct_iface_addr_t;
typedef struct uct_ep_addr       uct_ep_addr_t;
typedef struct uct_ep_params     uct_ep_params_t;
typedef struct uct_cm_attr       uct_cm_attr_t;
typedef struct uct_cm_params     uct_cm_params_t;
typedef struct uct_cm            uct_cm_t;
typedef uct_cm_t                 *uct_cm_h;
typedef struct uct_tag_context   uct_tag_context_t;
typedef uint64_t                 uct_tag_t;  /* tag type - 64 bit */
typedef int                      uct_worker_cb_id_t;
typedef void*                    uct_conn_request_h;
/**
 * @}
 */


/**
 * @ingroup UCT_RESOURCE
 * @brief Structure for scatter-gather I/O.
 *
 * Specifies a list of buffers which can be used within a single data transfer
 * function call.
 *
   @verbatim
    buffer
    |
    +-----------+-------+-----------+-------+-----------+
    |  payload  | empty |  payload  | empty |  payload  |
    +-----------+-------+-----------+-------+-----------+
    |<-length-->|       |<-length-->|       |<-length-->|
    |<---- stride ----->|<---- stride ----->|
   @endverbatim
 *
 * @note The sum of lengths in all iov list must be less or equal to max_zcopy
 *       of the respective communication operation.
 * @note If @a length or @a count are zero, the memory pointed to by @a buffer
 *       will not be accessed. Otherwise, @a buffer must point to valid memory.
 *
 * @note If @a count is one, every iov entry specifies a single contiguous data block
 *
 * @note If @a count > 1, each iov entry specifies a strided block of @a count
 *       elements and distance of @a stride byte between consecutive elements
 *
 */
typedef struct uct_iov {
    void     *buffer;   /**< Data buffer */
    size_t    length;   /**< Length of the payload in bytes */
    uct_mem_h memh;     /**< Local memory key descriptor for the data */
    size_t    stride;   /**< Stride between beginnings of payload elements in
                             the buffer in bytes */
    unsigned  count;    /**< Number of payload elements in the buffer */
} uct_iov_t;


/**
 * @ingroup UCT_AM
 * @brief Callback to process incoming active message
 *
 * When the callback is called, @a flags indicates how @a data should be handled.
 * If @a flags contain @ref UCT_CB_PARAM_FLAG_DESC value, it means @a data is part of
 * a descriptor which must be released later by @ref uct_iface_release_desc by
 * the user if the callback returns @ref UCS_INPROGRESS.
 *
 * @param [in]  arg      User-defined argument.
 * @param [in]  data     Points to the received data. This may be a part of
 *                       a descriptor which may be released later.
 * @param [in]  length   Length of data.
 * @param [in]  flags    Mask with @ref uct_cb_param_flags
 *
 * @note This callback could be set and released
 *       by @ref uct_iface_set_am_handler function.
 *
 * @retval UCS_OK         - descriptor was consumed, and can be released
 *                          by the caller.
 * @retval UCS_INPROGRESS - descriptor is owned by the callee, and would be
 *                          released later. Supported only if @a flags contain
 *                          @ref UCT_CB_PARAM_FLAG_DESC value. Otherwise, this is
 *                          an error.
 *
 */
typedef ucs_status_t (*uct_am_callback_t)(void *arg, void *data, size_t length,
                                          unsigned flags);


/**
 * @ingroup UCT_AM
 * @brief Callback to trace active messages.
 *
 * Writes a string which represents active message contents into 'buffer'.
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
 * @ingroup UCT_RESOURCE
 * @brief Callback to process send completion.
 *
 * @param [in]  self     Pointer to relevant completion structure, which was
 *                       initially passed to the operation.
 * @param [in]  status   Status of send action, possibly indicating an error.
 */
typedef void (*uct_completion_callback_t)(uct_completion_t *self,
                                          ucs_status_t status);


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback to process pending requests.
 *
 * @param [in]  self     Pointer to relevant pending structure, which was
 *                       initially passed to the operation.
 *
 * @return @ref UCS_OK         - This pending request has completed and
 *                               should be removed.
 *         @ref UCS_INPROGRESS - Some progress was made, but not completed.
 *                               Keep this request and keep processing the queue.
 *         Otherwise           - Could not make any progress. Keep this pending
 *                               request on the queue, and stop processing the queue.
 */
typedef ucs_status_t (*uct_pending_callback_t)(uct_pending_req_t *self);


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback to process peer failure.
 *
 * @param [in]  arg      User argument to be passed to the callback.
 * @param [in]  ep       Endpoint which has failed. Upon return from the callback,
 *                       this @a ep is no longer usable and all subsequent
 *                       operations on this @a ep will fail with the error code
 *                       passed in @a status.
 * @param [in]  status   Status indicating error.
 *
 * @return @ref UCS_OK   - The error was handled successfully.
 *         Otherwise     - The error was not handled and is returned back to
 *                         the transport.
 */
typedef ucs_status_t (*uct_error_handler_t)(void *arg, uct_ep_h ep,
                                            ucs_status_t status);


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback to purge pending requests.
 *
 * @param [in]  self     Pointer to relevant pending structure, which was
 *                       initially passed to the operation.
 * @param [in]  arg      User argument to be passed to the callback.
 */
typedef void (*uct_pending_purge_callback_t)(uct_pending_req_t *self,
                                             void *arg);

/**
 * @ingroup UCT_RESOURCE
 * @brief Callback for producing data.
 *
 * @param [in]  dest     Memory buffer to pack the data to.
 * @param [in]  arg      Custom user-argument.
 *
 * @return  Size of the data was actually produced.
 */
typedef size_t (*uct_pack_callback_t)(void *dest, void *arg);


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback for consuming data.
 *
 * @param [in]  arg      Custom user-argument.
 * @param [in]  data     Memory buffer to unpack the data from.
 * @param [in]  length   How much data to consume (size of "data")
 *
 * @note The arguments for this callback are in the same order as libc's memcpy().
 */
typedef void (*uct_unpack_callback_t)(void *arg, const void *data, size_t length);


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback to process an incoming connection request message on the server
 *        side.
 *
 * This callback routine will be invoked on the server side upon receiving an
 * incoming connection request. It should be set by the server side while
 * initializing an interface.
 * Incoming data is placed inside the conn_priv_data buffer.
 * This callback has to be thread safe.
 * Other than communication progress routines, it is allowed to call other UCT
 * communication routines from this callback.
 *
 * @param [in]  iface            Transport interface.
 * @param [in]  arg              User defined argument for this callback.
 * @param [in]  conn_request     Transport level connection request. The user
 *                               should accept or reject the request by calling
 *                               @ref uct_iface_accept or @ref uct_iface_reject
 *                               routines respectively.
 * @param [in]  conn_priv_data   Points to the received data.
 *                               This is the private data that was passed to the
 *                               @ref uct_ep_params_t::sockaddr_pack_cb on the
 *                               client side.
 * @param [in]  length           Length of the received data.
 *
 */
typedef void
(*uct_sockaddr_conn_request_callback_t)(uct_iface_h iface, void *arg,
                                        uct_conn_request_h conn_request,
                                        const void *conn_priv_data,
                                        size_t length);


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback to fill the user's private data on the client side.
 *
 * This callback routine will be invoked on the client side before sending the
 * transport's connection request to the server.
 * The callback routine must be set by the client when creating an endpoint.
 * The user's private data should be placed inside the priv_data buffer to be
 * sent to the server side.
 * The maximal allowed length of the private data is indicated by the field
 * max_conn_priv inside @ref uct_iface_attr.
 * Communication progress routines should not be called from this callback.
 * It is allowed to call other UCT communication routines from this callback.
 *
 * @param [in]  arg        User defined argument for this callback.
 * @param [in]  dev_name   Device name. This routine may fill the user's private
 *                         data according to the given device name.
 *                         The device name that is passed to this routine,
 *                         corresponds to the dev_name field inside
 *                         @ref uct_tl_resource_desc_t as returned from
 *                         @ref uct_md_query_tl_resources.
 * @param [out] priv_data  User's private data to be passed to the server side.
 *
 * @return Negative value indicates an error according to @ref ucs_status_t.
 *         On success, non-negative value indicates actual number of
 *         bytes written to the @a priv_data buffer.
 */
typedef ssize_t (*uct_sockaddr_priv_pack_callback_t)(void *arg,
                                                     const char *dev_name,
                                                     void *priv_data);


/**
 * @ingroup UCT_TAG
 * @brief Callback to process unexpected eager tagged message.
 *
 * This callback is invoked when tagged message sent by eager protocol has
 * arrived and no corresponding tag has been posted.
 *
 * @note The callback is always invoked from the context (thread, process)
 *       that called @a uct_iface_progress().
 *
 * @note It is allowed to call other communication routines from the callback.
 *
 * @param [in]  arg     User-defined argument
 * @param [in]  data    Points to the received unexpected data.
 * @param [in]  length  Length of data.
 * @param [in]  desc    Points to the received descriptor, at the beginning of
 *                      the user-defined rx_headroom.
 * @param [in]  stag    Tag from sender.
 * @param [in]  imm     Immediate data from sender.
 *
 * @warning If the user became the owner of the @a desc (by returning
 *          @ref UCS_INPROGRESS) the descriptor must be released later by
 *          @ref uct_iface_release_desc by the user.
 *
 * @retval UCS_OK         - descriptor was consumed, and can be released
 *                          by the caller.
 * @retval UCS_INPROGRESS - descriptor is owned by the callee, and would be
 *                          released later.
 */
typedef ucs_status_t (*uct_tag_unexp_eager_cb_t)(void *arg, void *data,
                                                 size_t length, unsigned flags,
                                                 uct_tag_t stag, uint64_t imm);


/**
 * @ingroup UCT_TAG
 * @brief Callback to process unexpected rendezvous tagged message.
 *
 * This callback is invoked when rendezvous send notification has arrived
 * and no corresponding tag has been posted.
 *
 * @note The callback is always invoked from the context (thread, process)
 *       that called @a uct_iface_progress().
 *
 * @note It is allowed to call other communication routines from the callback.
 *
 * @param [in]  arg           User-defined argument
 * @param [in]  flags         Mask with @ref uct_cb_param_flags
 * @param [in]  stag          Tag from sender.
 * @param [in]  header        User defined header.
 * @param [in]  header_length User defined header length in bytes.
 * @param [in]  remote_addr   Sender's buffer virtual address.
 * @param [in]  length        Sender's buffer length.
 * @param [in]  rkey_buf      Sender's buffer packed remote key. It can be
 *                            passed to uct_rkey_unpack() to create uct_rkey_t.
 *
 * @warning If the user became the owner of the @a desc (by returning
 *          @ref UCS_INPROGRESS) the descriptor must be released later by
 *          @ref uct_iface_release_desc by the user.
 *
 * @retval UCS_OK         - descriptor was consumed, and can be released
 *                          by the caller.
 * @retval UCS_INPROGRESS - descriptor is owned by the callee, and would be
 *                          released later.
 */
typedef ucs_status_t (*uct_tag_unexp_rndv_cb_t)(void *arg, unsigned flags,
                                                uint64_t stag, const void *header,
                                                unsigned header_length,
                                                uint64_t remote_addr, size_t length,
                                                const void *rkey_buf);


#endif
