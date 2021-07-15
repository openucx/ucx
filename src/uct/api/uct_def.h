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


#define UCT_COMPONENT_NAME_MAX     16
#define UCT_TL_NAME_MAX            10
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
 *
 * If UCT_CB_PARAM_FLAG_DESC flag is enabled, then data is part of a descriptor
 * which includes the user-defined rx_headroom, and the callback may return
 * UCS_INPROGRESS and hold on to that descriptor. Otherwise, the data can't be
 * used outside the callback. If needed, the data must be copied-out.
 *
   @verbatim
    descriptor    data
    |             |
    +-------------+-------------------------+
    | rx_headroom | payload                 |
    +-------------+-------------------------+
   @endverbatim
 *
 * UCT_CB_PARAM_FLAG_FIRST and UCT_CB_PARAM_FLAG_MORE flags are relevant for
 * @ref uct_tag_unexp_eager_cb_t callback only. The former value indicates that
 * the data is the first fragment of the message. The latter value means that
 * more fragments of the message yet to be delivered.
 */
enum uct_cb_param_flags {
    UCT_CB_PARAM_FLAG_DESC  = UCS_BIT(0),
    UCT_CB_PARAM_FLAG_FIRST = UCS_BIT(1),
    UCT_CB_PARAM_FLAG_MORE  = UCS_BIT(2)
};

/**
 * @addtogroup UCT_RESOURCE
 * @{
 */
typedef struct uct_component         *uct_component_h;
typedef struct uct_iface             *uct_iface_h;
typedef struct uct_iface_config      uct_iface_config_t;
typedef struct uct_md_config         uct_md_config_t;
typedef struct uct_cm_config         uct_cm_config_t;
typedef struct uct_ep                *uct_ep_h;
typedef void *                       uct_mem_h;
typedef uintptr_t                    uct_rkey_t;
typedef struct uct_md                *uct_md_h;          /**< @brief Memory domain handler */
typedef struct uct_md_ops            uct_md_ops_t;
typedef void                         *uct_rkey_ctx_h;
typedef struct uct_iface_attr        uct_iface_attr_t;
typedef struct uct_iface_params      uct_iface_params_t;
typedef struct uct_ep_attr           uct_ep_attr_t;
typedef struct uct_md_attr           uct_md_attr_t;
typedef struct uct_completion        uct_completion_t;
typedef struct uct_pending_req       uct_pending_req_t;
typedef struct uct_worker            *uct_worker_h;
typedef struct uct_md                uct_md_t;
typedef enum uct_am_trace_type       uct_am_trace_type_t;
typedef struct uct_device_addr       uct_device_addr_t;
typedef struct uct_iface_addr        uct_iface_addr_t;
typedef struct uct_ep_addr           uct_ep_addr_t;
typedef struct uct_ep_params         uct_ep_params_t;
typedef struct uct_ep_connect_params uct_ep_connect_params_t;
typedef struct uct_cm_attr           uct_cm_attr_t;
typedef struct uct_cm                uct_cm_t;
typedef uct_cm_t                     *uct_cm_h;
typedef struct uct_listener_attr     uct_listener_attr_t;
typedef struct uct_listener          *uct_listener_h;
typedef struct uct_listener_params   uct_listener_params_t;
typedef struct uct_tag_context       uct_tag_context_t;
typedef uint64_t                     uct_tag_t;  /* tag type - 64 bit */
typedef int                          uct_worker_cb_id_t;
typedef void*                        uct_conn_request_h;

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
 * @ingroup UCT_CLIENT_SERVER
 * @brief Client-Server private data pack callback arguments field mask.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_cm_ep_priv_data_pack_args are present, for backward compatibility
 * support.
 */
enum uct_cm_ep_priv_data_pack_args_field {
    /** Enables @ref uct_cm_ep_priv_data_pack_args::dev_name
     *  Indicates that dev_name field in uct_cm_ep_priv_data_pack_args_t is
     *  valid.
     */
    UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME = UCS_BIT(0)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Client-Server resolve callback arguments field mask.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_cm_ep_resolve_args are present, for backward compatibility support.
 */
enum uct_cm_ep_resolve_args_field {
    /**
     * Indicates that @ref uct_cm_ep_resolve_args::dev_name is valid.
     */
    UCT_CM_EP_RESOLVE_ARGS_FIELD_DEV_NAME       = UCS_BIT(0)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Arguments to the client-server private data pack callback.
 *
 * Used with the client-server API on a connection manager.
 */
typedef struct uct_cm_ep_priv_data_pack_args {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_cm_ep_priv_data_pack_args_field.
     * Fields not specified by this mask should not be accessed by the callback.
     */
    uint64_t                    field_mask;

    /**
     * Device name. This routine may fill the user's private data according to
     * the given device name. The device name that is passed to this routine,
     * corresponds to @ref uct_tl_resource_desc_t::dev_name as returned from
     * @ref uct_md_query_tl_resources.
     */
    char                        dev_name[UCT_DEVICE_NAME_MAX];
} uct_cm_ep_priv_data_pack_args_t;


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Arguments to the client-server resolved callback.
 *
 * Used with the client-server API on a connection manager.
 */
typedef struct uct_cm_ep_resolve_args {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_cm_ep_resolve_args_field.
     * Fields not specified by this mask should not be accessed by the callback.
     */
    uint64_t                    field_mask;

   /**
     * Device name indicates the device that the endpoint was bound to during
     * address and route resolution. The device name that is passed to this
     * callback, corresponds to @ref uct_tl_resource_desc_t::dev_name as
     * returned from @ref uct_md_query_tl_resources.
     */
    char                        dev_name[UCT_DEVICE_NAME_MAX];
} uct_cm_ep_resolve_args_t;


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Remote data attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref uct_cm_remote_data are
 * present, for backward compatibility support.
 */
enum uct_cm_remote_data_field {
    /** Enables @ref uct_cm_remote_data::dev_addr */
    UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR              = UCS_BIT(0),

    /** Enables @ref uct_cm_remote_data::dev_addr_length */
    UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH       = UCS_BIT(1),

    /** Enables @ref uct_cm_remote_data::conn_priv_data */
    UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA        = UCS_BIT(2),

    /** Enables @ref uct_cm_remote_data::conn_priv_data_length */
    UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH = UCS_BIT(3)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Data received from the remote peer.
 *
 * The remote peer's device address, the data received from it and their lengths.
 * Used with the client-server API on a connection manager.
 */
typedef struct uct_cm_remote_data {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_cm_remote_data_field. Fields not specified by this mask
     * will be ignored.
     */
    uint64_t                field_mask;

    /**
     * Device address of the remote peer.
     */
    const uct_device_addr_t *dev_addr;

    /**
     * Length of the remote device address.
     */
    size_t                  dev_addr_length;

    /**
     * Pointer to the received data. This is the private data that was passed to
     * @ref uct_ep_params_t::sockaddr_pack_cb.
     */
    const void              *conn_priv_data;

    /**
     * Length of the received data from the peer.
     */
    size_t                  conn_priv_data_length;
} uct_cm_remote_data_t;


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Listener's connection request callback arguments field mask.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_cm_listener_conn_request_args are present, for backward compatibility
 * support.
 */
enum uct_cm_listener_conn_request_args_field {
    /** Enables @ref uct_cm_listener_conn_request_args::dev_name
     *  Indicates that dev_name field in uct_cm_listener_conn_request_args_t is
     *  valid.
     */
    UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_DEV_NAME     = UCS_BIT(0),

    /** Enables @ref uct_cm_listener_conn_request_args::conn_request
     *  Indicates that conn_request field in uct_cm_listener_conn_request_args_t
     *  is valid.
     */
    UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CONN_REQUEST = UCS_BIT(1),

    /** Enables @ref uct_cm_listener_conn_request_args::remote_data
     *  Indicates that remote_data field in uct_cm_listener_conn_request_args_t
     *  is valid.
     */
    UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_REMOTE_DATA  = UCS_BIT(2),

    /** Enables @ref uct_cm_listener_conn_request_args::client_address
     *  Indicates that client_address field in uct_cm_listener_conn_request_args_t
     *  is valid.
     */
    UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CLIENT_ADDR  = UCS_BIT(3)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Arguments to the listener's connection request callback.
 *
 * The local device name, connection request handle and the data the client sent.
 * Used with the client-server API on a connection manager.
 */
typedef struct uct_cm_listener_conn_request_args {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_cm_listener_conn_request_args_field.
     * Fields not specified by this mask should not be accessed by the callback.
     */
    uint64_t                   field_mask;

    /**
     * Local device name which handles the incoming connection request.
     */
    char                       dev_name[UCT_DEVICE_NAME_MAX];

    /**
     * Connection request handle. Can be passed to this callback from the
     * transport and will be used by it to accept or reject the connection
     * request from the client.
     */
    uct_conn_request_h         conn_request;

    /**
     * Remote data from the client.
     */
    const uct_cm_remote_data_t *remote_data;

    /**
     * Client's address.
     */
    ucs_sock_addr_t            client_address;
} uct_cm_listener_conn_request_args_t;


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Field mask flags for client-side connection established callback.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_cm_ep_client_connect_args are present, for backward compatibility
 * support.
 */
enum uct_cm_ep_client_connect_args_field {
    /** Enables @ref uct_cm_ep_client_connect_args::remote_data */
    UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_REMOTE_DATA = UCS_BIT(0),

    /** Enables @ref uct_cm_ep_client_connect_args::status */
    UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_STATUS      = UCS_BIT(1)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Arguments to the client's connect callback.
 *
 * Used with the client-server API on a connection manager.
 */
typedef struct uct_cm_ep_client_connect_args {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_cm_ep_client_connect_args_field.
     * Fields not specified by this mask should not be accessed by the callback.
     */
    uint64_t                   field_mask;

    /**
     * Remote data from the server.
     */
    const uct_cm_remote_data_t *remote_data;

    /**
     * Indicates the connection establishment response from the remote server:
     * UCS_OK                   - the remote server accepted the connection request.
     * UCS_ERR_REJECTED         - the remote server rejected the connection request.
     * UCS_ERR_CONNECTION_RESET - the server's connection was reset during
     *                            the connection establishment to the client.
     * Otherwise                - indicates an internal connection establishment
     *                            error on the local (client) side.
     */
    ucs_status_t               status;
} uct_cm_ep_client_connect_args_t;


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Field mask flags for server-side connection established notification
 *        callback.
 *
 * The enumeration allows specifying which fields in
 * @ref uct_cm_ep_server_conn_notify_args are present, for backward compatibility
 * support.
 */
enum uct_cm_ep_server_conn_notify_args_field {
    /** Enables @ref uct_cm_ep_server_conn_notify_args::status
     *  Indicates that status field in uct_cm_ep_server_conn_notify_args_t is valid.
     */
    UCT_CM_EP_SERVER_CONN_NOTIFY_ARGS_FIELD_STATUS = UCS_BIT(0)
};


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Arguments to the server's notify callback.
 *
 * Used with the client-server API on a connection manager.
 */
typedef struct uct_cm_ep_server_conn_notify_args {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref uct_cm_ep_server_conn_notify_args_field.
     * Fields not specified by this mask should not be accessed by the callback.
     */
    uint64_t                   field_mask;

    /**
     * Indicates the client's @ref ucs_status_t status:
     * UCS_OK                   - the client completed its connection
     *                            establishment and called
     *                            @ref uct_cm_client_ep_conn_notify
     * UCS_ERR_CONNECTION_RESET - the client's connection was reset during
     *                            the connection establishment to the server.
     * Otherwise                - indicates an internal connection establishment
     *                            error on the local (server) side.
     */
    ucs_status_t               status;
} uct_cm_ep_server_conn_notify_args_t;


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
 */
typedef void (*uct_completion_callback_t)(uct_completion_t *self);


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
 * @ingroup UCT_CLIENT_SERVER
 * @brief Callback to process an incoming connection request on the server side.
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
 *                               conn_request should not be used outside the
 *                               scope of this callback.
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
 * @ingroup UCT_CLIENT_SERVER
 * @brief Callback to process an incoming connection request on the server side
 *        listener in a connection manager.
 *
 * This callback routine will be invoked on the server side upon receiving an
 * incoming connection request. It should be set by the server side while
 * initializing a listener in a connection manager.
 * This callback has to be thread safe.
 * Other than communication progress routines, it is allowed to call other UCT
 * communication routines from this callback.
 *
 * @param [in]  listener         Transport listener.
 * @param [in]  arg              User argument for this callback as defined in
 *                               @ref uct_listener_params_t::user_data
 * @param [in]  conn_req_args    Listener's arguments to handle the connection
 *                               request from the client.
 */
typedef void
(*uct_cm_listener_conn_request_callback_t)(uct_listener_h listener, void *arg,
                                           const uct_cm_listener_conn_request_args_t
                                           *conn_req_args);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Callback to process an incoming connection establishment acknowledgment
 *        on the server side listener, from the client, which indicates that the
 *        client side is connected.
 *        The callback also notifies the server side of a local error on a
 *        not-yet-connected endpoint.
 *
 * This callback routine will be invoked on the server side upon receiving an
 * incoming connection establishment acknowledgment from the client, which is sent
 * from it once the client is connected to the server. Used to connect the server
 * side to the client or handle an error from it - depending on the status field.
 * This callback will also be invoked in the event of an internal local error
 * with a failed @ref uct_cm_ep_server_conn_notify_args::status if the endpoint
 * was not connected yet.
 * This callback has to be thread safe.
 * Other than communication progress routines, it is permissible to call other UCT
 * communication routines from this callback.
 *
 * @param [in]  ep               Transport endpoint.
 * @param [in]  arg              User argument for this callback as defined in
 *                               @ref uct_ep_params_t::user_data
 * @param [in]  connect_args     Server's connect callback arguments.
 */
typedef void (*uct_cm_ep_server_conn_notify_callback_t)
                (uct_ep_h ep, void *arg,
                 const uct_cm_ep_server_conn_notify_args_t *connect_args);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Callback to process an incoming connection response on the client side
 *        from the server or handle a local error on a not-yet-connected endpoint.
 *
 * This callback routine will be invoked on the client side upon receiving an
 * incoming connection response from the server. Used to connect the client side
 * to the server or handle an error from it - depending on the status field.
 * This callback will also be invoked in the event of an internal local error
 * with a failed @ref uct_cm_ep_client_connect_args::status if the endpoint was
 * not connected yet.
 * This callback has to be thread safe.
 * Other than communication progress routines, it is permissible to call other UCT
 * communication routines from this callback.
 *
 * @param [in]  ep               Transport endpoint.
 * @param [in]  arg              User argument for this callback as defined in
 *                               @ref uct_ep_params_t::user_data.
 * @param [in]  connect_args     Client's connect callback arguments
 */
typedef void (*uct_cm_ep_client_connect_callback_t)(uct_ep_h ep, void *arg,
                                                    const uct_cm_ep_client_connect_args_t
                                                    *connect_args);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Callback to handle the disconnection of the remote peer.
 *
 * This callback routine will be invoked on the client and server sides upon
 * a disconnect of the remote peer. It will disconnect the given endpoint from
 * the remote peer.
 * This callback won't be invoked if the endpoint was not connected to the remote
 * peer yet.
 * This callback has to be thread safe.
 * Other than communication progress routines, it is permissible to call other UCT
 * communication routines from this callback.
 *
 * @param [in]  ep               Transport endpoint to disconnect.
 * @param [in]  arg              User argument for this callback as defined in
 *                               @ref uct_ep_params_t::user_data.
 */
typedef void (*uct_ep_disconnect_cb_t)(uct_ep_h ep, void *arg);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Callback to fill the user's private data in a client-server flow.
 *
 * This callback routine will be invoked on the client side, before sending the
 * transport's connection request to the server, or on the server side before
 * sending a connection response to the client.
 * This callback routine can be set when creating an endpoint.
 * The user's private data should be placed inside the priv_data buffer to be
 * sent to the remote side.
 * The maximal allowed length of the private data is indicated by the field
 * max_conn_priv inside @ref uct_iface_attr or inside @ref uct_cm_attr when using a
 * connection manager.
 * Communication progress routines should not be called from this callback.
 * It is allowed to call other UCT communication routines from this callback.
 *
 * @param [in]  arg          User defined argument for this callback.
 * @param [in]  pack_args    Handle for the the private data packing.
 * @param [out] priv_data    User's private data to be passed to the remote side.
 *
 * @return Negative value indicates an error according to @ref ucs_status_t.
 *         On success, a non-negative value indicates actual number of
 *         bytes written to the @a priv_data buffer.
 */
typedef ssize_t
(*uct_cm_ep_priv_data_pack_callback_t)(void *arg,
                                       const uct_cm_ep_priv_data_pack_args_t
                                       *pack_args, void *priv_data);


/**
 * @ingroup UCT_CLIENT_SERVER
 * @brief Callback to notify that the client side endpoint is bound to a
 *        local device.
 *
 * This callback routine will be invoked, when the client side endpoint is bound
 * to a local device.
 * The callback routine can be set when creating an endpoint.
 * Communication progress routines should not be called from this callback.
 * It is allowed to call other UCT communication routines from this callback.
 *
 * @param [in]  user_data       User argument as defined in
 *                              @ref uct_ep_params_t::user_data.
 * @param [in]  resolve_args    Handle for the extra arguments provided by the
 *                              transport.
 *
 * @return UCS_OK on success or error as defined in @ref ucs_status_t.
 */
typedef ucs_status_t
(*uct_cm_ep_resolve_callback_t)(void *user_data,
                                const uct_cm_ep_resolve_args_t *resolve_args);


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
 * @param [in]     arg     User-defined argument
 * @param [in]     data    Points to the received unexpected data.
 * @param [in]     length  Length of data.
 * @param [in]     flags   Mask with @ref uct_cb_param_flags flags. If it
 *                         contains @ref UCT_CB_PARAM_FLAG_DESC value, this means
 *                         @a data is part of a descriptor which must be released
 *                         later using @ref uct_iface_release_desc by the user if
 *                         the callback returns @ref UCS_INPROGRESS.
 * @param [in]     stag    Tag from sender.
 * @param [in]     imm     Immediate data from sender.
 *
 * @param [inout]  context Storage for a per-message user-defined context. In
 *                         this context, the message is defined by the sender
 *                         side as a single call to uct_ep_tag_eager_short/bcopy/zcopy.
 *                         On the transport level the message can be fragmented
 *                         and delivered to the target over multiple fragments.
 *                         The fragments will preserve the original order of the
 *                         message. Each fragment will result in invocation of
 *                         the above callback. The user can use
 *                         UCT_CB_PARAM_FLAG_FIRST to identify the first fragment,
 *                         allocate the context object and use the context as a
 *                         token that is set by the user and passed to subsequent
 *                         callbacks of the same message. The user is responsible
 *                         for allocation and release of the context.
 *
 * @note No need to allocate the context in the case of a single fragment message
 *       (i.e. @a flags contains @ref UCT_CB_PARAM_FLAG_FIRST, but does not
 *       contain @ref UCT_CB_PARAM_FLAG_MORE).
 *
 * @retval UCS_OK          - data descriptor was consumed, and can be released
 *                           by the caller.
 * @retval UCS_INPROGRESS  - data descriptor is owned by the callee, and will be
 *                           released later.
 */
typedef ucs_status_t (*uct_tag_unexp_eager_cb_t)(void *arg, void *data,
                                                 size_t length, unsigned flags,
                                                 uct_tag_t stag, uint64_t imm,
                                                 void **context);


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


/**
 * @ingroup UCT_RESOURCE
 * @brief Callback to process asynchronous events.
 *
 * @param [in]  arg      User argument to be passed to the callback.
 * @param [in]  flags    Flags to be passed to the callback (reserved for
 *                       future use).
 */
typedef void (*uct_async_event_cb_t)(void *arg, unsigned flags);


#endif
