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
 * @brief UCP active message rendezvous parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_params_t are
 * present. It is used to enable backward compatibility support.
 */
enum ucp_am_rendezvous_params_field {
  /* Max size of the iovec returned by the initial AM */
  UCP_AM_RENDEZVOUS_FIELD_IOVEC_SIZE = UCS_BIT(0)
};

/**
 * @brief Tuning parameters for the UCP active message rendezvous.
 *
 * The structure defines the parameters that are used for the
 * UCP active message rendezvous function
 */
typedef struct ucp_am_rendezvous_params {
  uint64_t field_mask ;
  /* Number of elements that the ucp library should allocate in the 'iovec'
   * for the rendezvous callback to fill in.
   */
  size_t iovec_size ;
} ucp_am_rendezvous_params_t ;

/*
 * The ucp library will drive a function on the arrival of each fragment
 * of data from the client.
 */
typedef void (*ucp_am_data_function_t)(
                                       void *target,
                                       void *source,
                                       size_t bytes,
                                       void *cookie
                                      ) ;
/*
 * The ucp library will drive a function when the transfer from the
 * client is complete.
 */
typedef ucs_status_t (*ucp_am_local_function_t)(
                                                void *arg,
                                                void *cookie,
                                                ucp_dt_iov_t *iovec,
                                                size_t iovec_length
                                               ) ;

/*
 * Structure for communicatiob between the active message rendezvous
 * callback and the ucp library. The ucp library sets up 'iovec_max_length'
 * to indicate the length of the iovec in this structure; the remaining
 * fields are set by the rendezvous callback to indicate what should happen
 * with the remaining data from the client. In the initial implementation,
 * iovec_max_length is always 1.
 */
typedef struct ucp_am_rendezvous_recv {
    /* Function to be driven when data transfer is complete */
    ucp_am_local_function_t local_fn;
    /* Argument to be passed to local_fn */
    void                   *cookie;
    /* Function to be driven when each fragment of data transfer is complete
     * In the initial implementation, all data is transferred in one fragment, so
     * the data_fn si driven once just before the local_fn
     */
    ucp_am_data_function_t  data_fn;
    /* Argument to be passed to data_fn */
    void                   *data_cookie ;
    /* Sise of iovec allocated by the ucp library */
    size_t                  iovec_max_length;
    /* Size of iovec filled in by the rendezvous callback */
    size_t                  iovec_length;
    /* iovec indicating where the data from the client is to be placed */
    ucp_dt_iov_t            iovec[1];
  } ucp_am_rendezvous_recv_t;

  /**
   * @ingroup UCP_ENDPOINT
   * @brief Callback to process incoming Active Message rendezvous.
   *
   * When the callback is called, @a flags indicates how @a data should be handled.
   *
   * @param [in]  arg      User-defined argument.
   * @param [in]  data     Points to the received data header.
   * @param [in]  length   Length of data header.
   * @param [in]  reply_ep If the Active Message is sent with the
   *                       UCP_AM_SEND_REPLY flag, the sending ep
   *                       will be passed in. If not, NULL will be passed.
   * @param [in]  flags    0.
   * @parame[out] recv     Struture passed to the ucp library to control
   *                       placement of the remaining data
   *
   * @return UCS_OK        @a data will not persist after the callback returns.
   *
   *
   * @note This callback should be set and released
   *       by @ref ucp_worker_set_am_rendezvous handler function.
   *
   */
typedef ucs_status_t (*ucp_am_rendezvous_callback_t)(
                                                 void *arg,
                                                 void *data,
                                                 size_t length,
                                                 ucp_ep_h reply_ep,
                                                 unsigned flags,
                                                 size_t remaining_length,
                                                 ucp_am_rendezvous_recv_t *recv
                                                    );

/**
 * @ingroup UCP_WORKER
 * @brief Add user defined callback for Active Message rendezvous.
 *
 * This routine installs a user defined callback to handle incoming Active
 * Messages with a specific id. This callback is called whenever an Active
 * Message that was sent from the remote peer by @ref ucp_am_rendezvous_send_nb
 * is received on this worker, if the ucp library decides that rendezvous
 * processing is appropriate for this active message; otherwise the callback
 * registered with ucp_worker_set_am_handler will be driven.
 *
 * @param [in]  worker      UCP worker on which to set the Active Message
 *                          handler.
 * @param [in]  id          Active Message id.
 * @param [in]  cb          Active Message callback. NULL to clear.
 * @param [in]  arg         Active Message argument, which will be passed
 *                          in to every invocation of the callback as the
 *                          arg argument.
 * @param [in]  flags       Dictates how an Active Message is handled on the
 *                          remote endpoint. No flags currently defined.
 * @param [in]  params      Tuning parameters for the active message
 *
 * @return error code if the worker does not support Active Messages or
 *         requested callback flags.
 */
ucs_status_t ucp_worker_set_am_rendezvous_handler(
                                       ucp_worker_h worker,
                                       uint16_t id,
                                       ucp_am_rendezvous_callback_t cb,
                                       void *arg,
                                       uint32_t flags,
                                       const ucp_am_rendezvous_params_t *params
                                       );

/**
 * @ingroup UCP_COMM
 * @brief Send Active Message with transfer using RENDEZVOUS
 *
 * This routine sends an Active Message to an ep. It does not support
 * CUDA memory.
 *
 * @param [in]  ep          UCP endpoint where the Active Message will be run.
 * @param [in]  id          Active Message id. Specifies which registered
 *                          callback to run.
 * @param [in]  buffer      Pointer to the data to be sent to the target node
 *                          of the Active Message.
 * @param [in]  count       Number of elements to send.
 * @param [in]  datatype    Datatype descriptor for the elements in the buffer.
 * @param [in]  cb          Callback that is invoked upon completion of the
 *                          data transfer if it is not completed immediately.
 * @param [in]  flags       For Future use.
 *
 * @return UCS_OK           Active Message was sent immediately.
 * @return UCS_PTR_IS_ERR(_ptr) Error sending Active Message.
 * @return otherwise        Pointer to request, and Active Message is known
 *                          to be completed after cb is run.
 */
ucs_status_ptr_t ucp_am_rendezvous_send_nb(ucp_ep_h ep, uint16_t id,
                                const void *buffer, size_t count,
                                ucp_datatype_t datatype,
                                ucp_send_callback_t cb, unsigned flags);

END_C_DECLS

#endif
