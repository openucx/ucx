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

enum ucp_am_rendezvous_params_field {
  UCP_AM_RENDEZVOUS_FIELD_IOVEC_SIZE = UCS_BIT(0)  /* Max size of the iovec returned by the initial AM */
};

typedef struct ucp_am_rendezvous_params {
  uint64_t field_mask ;
  size_t iovec_size ;
} ucp_am_rendezvous_params_t ;


typedef void (*ucp_am_data_function_t)(void *target, void *source, size_t bytes, void *cookie) ;
typedef ucs_status_t (*ucp_am_local_function_t)(void *arg, void *cookie, ucp_dt_iov_t *iovec, size_t iovec_length) ;

typedef struct ucp_am_rendezvous_recv {
    ucp_am_local_function_t local_fn ;
    void * cookie ;
    ucp_am_data_function_t data_fn ;
    void *data_cookie ;
    size_t iovec_max_length ;
    size_t iovec_length ;
    ucp_dt_iov_t iovec[1] ;
  } ucp_am_rendezvous_recv_t ;

typedef ucs_status_t (*ucp_am_rendezvous_callback_t)(void *arg, void *data, size_t length,
                                          ucp_ep_h reply_ep, unsigned flags, size_t remaining_length, ucp_am_rendezvous_recv_t *recv);

ucs_status_t ucp_worker_set_am_rendezvous_handler(ucp_worker_h worker, uint16_t id,
                                       ucp_am_rendezvous_callback_t cb, void *arg,
                                       uint32_t flags, const ucp_am_rendezvous_params_t *params );

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
