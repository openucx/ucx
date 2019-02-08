/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCPX_H_
#define UCPX_H_

#include <ucp/api/ucp_def.h>
#include <ucs/sys/compiler_def.h>

/*
 * This header file is for experimental UCP API.
 * APIs defined here are NOT stable and may be removed / changed without notice.
 * By default, this header file is not installed. In order to install it, need
 * to run ./configure --enable-experimental-api
 */

BEGIN_C_DECLS

/**
 * @ingroup UCP_ENDPOINT
 * @brief Callback to process incoming active message
 *
 * When the callback is called, @a flags indicates how @a data should be handled.
 *  
 * @param [in]  arg      User-defined argument.
 * @param [in]  data     Points to the received data. This data may
 *                       persist after the callback returns and need
 *                       to be freed with @ref ucp_am_data_release
 * @param [in]  length   Length of data.
 * @param [in]  reply_ep If the active message is sent with the 
 *                       UCP_AM_SEND_REPLY flag, the sending ep
 *                       will be passed in. If not, NULL will be passed
 * @param [in]  flags    If this flag is set to UCP_CB_PARAM_FLAG_DATA,
 *                       the callback can return UCS_INPROGRESS and
 *                       data will persist after the callback returns
 *
 * @return UCS_OK        @a data will not persist after the callback returns
 *                      
 * @return UCS_INPROGRESS Can only be returned if flags is set to
 *                        UCP_CB_PARAM_FLAG_DATA. If UCP_INPROGRESS
 *                        is returned, data will persist after the
 *                        callback has returned. To free the memory,
 *                        a pointer to the data must be passed into
 *                        @ref ucp_am_data_release
 *
 * @note This callback could be set and released
 *       by @ref ucp_worker_set_am_handler function.
 *
 */
typedef ucs_status_t (*ucp_am_callback_t)(void *arg, void *data, size_t length,
                                          ucp_ep_h reply_ep, unsigned flags);


/**
 * @ingroup UCP_WORKER
 * @brief Flags for a UCP AM callback
 *
 * Flags that indicate how to handle UCP Active Messages
 * Currently only UCP_AM_FLAG_WHOLE_MSG is supported,
 * which indicates the entire message is handled in one
 * callback
 */
enum ucp_am_cb_flags {
    UCP_AM_FLAG_WHOLE_MSG = UCS_BIT(0)
};


/** 
 * @ingroup UCP_WORKER
 * @brief Flags for sending a UCP AM
 *
 * Flags dictate the behavior of ucp_am_send_nb
 * currently the only flag tells ucp to pass in
 * the sending endpoint to the call
 * back so a reply can be defined
 */
enum ucp_send_am_flags {
    UCP_AM_SEND_REPLY = UCS_BIT(0)
};


/**
 * @ingroup UCP_ENDPOINT
 * @brief Descriptor flags for Active Message Callback
 *
 * In a callback, if flags is set to UCP_CB_PARAM_FLAG_DATA, data
 * was allocated, so if UCS_INPROGRESS is returned from the
 * callback, the data parameter will persist and the user has to call
 * @ref ucp_am_data_release
 */
enum ucp_cb_param_flags {
    UCP_CB_PARAM_FLAG_DATA = UCS_BIT(0)
};


/**
 * @ingroup UCP_WORKER
 * @brief Add user defined callback for active message.
 *
 * This routine installs a user defined callback to handle incoming active
 * messages with a specific id. This callback is called whenever an active message,
 * which was sent from the remote peer by @ref for ucp_am_send_nb, is received on 
 * this worker.
 *
 * @param [in]  worker      UCP worker on which to set the am handler
 * @param [in]  id          Active message id.
 * @param [in]  cb          Active message callback. NULL to clear.
 * @param [in]  arg         Active message argument, which will be passed in to
 *                          every invocation of the callback as the arg argument.
 * @param [in]  flags       Dictates how an Active Message is handled on the remote endpoint.
 *                          Currently only UCP_AM_FLAG_WHOLE_MSG is supported, which indicates
 *                          the callback will not be invoked until all data has arrived.
 *
 * @return error code if the worker does not support active messages or 
 *         requested callback flags
 */
ucs_status_t ucp_worker_set_am_handler(ucp_worker_h worker, uint16_t id, 
                                       ucp_am_callback_t cb, void *arg,
                                       uint32_t flags);


/**
 * @ingroup UCP_COMM
 * @brief Send Active Message
 *
 * This routine sends an Active Message to an ep. It does not support
 * CUDA memory.
 *
 * @param [in]  ep          UCP endpoint where the active message will be run
 * @param [in]  id          Active Message id. Specifies which registered 
 *                          callback to run.
 * @param [in]  buffer      Pointer to the data to be sent to the target node 
 *                          for the AM.
 * @param [in]  count       Number of elements to send.
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer. 
 * @param [in]  cb          Callback that is invoked upon completion of the data
 *                          transfer if it is not completed immediately
 * @param [in]  flags       For Future use
 *
 * @return UCS_OK           Active message was sent immediately
 * @return UCS_PTR_IS_ERR(_ptr) Error sending Active Message
 * @return otherwise        Pointer to request, and Active Message is known
 *                          to be completed after cb is run
 */
ucs_status_ptr_t ucp_am_send_nb(ucp_ep_h ep, uint16_t id,
                                const void *buffer, size_t count,
                                ucp_datatype_t datatype,
                                ucp_send_callback_t cb, unsigned flags);


/**
 * @ingroup UCP_COMM
 * @brief Releases am data
 *
 * This routine releases back data that persisted through an AM
 * callback because that callback returned UCS_INPROGRESS
 *
 * @param [in] worker       Worker which received the active message
 * @param [in] data         Pointer to data that was passed into
 *                          the Active Message callback as the data
 *                          parameter and the callback flags were set to 
 *                          UCP_CB_PARAM_FLAG_DATA
 */
void ucp_am_data_release(ucp_worker_h worker, void *data);


END_C_DECLS

#endif
