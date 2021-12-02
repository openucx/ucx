/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SCOPY_EP_H
#define UCT_SCOPY_EP_H

#include <uct/base/uct_iface.h>
#include <uct/sm/base/sm_ep.h>
#include <ucs/sys/iovec.h>


extern const char* uct_scopy_tx_op_str[];


typedef enum uct_scopy_tx_op {
    UCT_SCOPY_TX_GET_ZCOPY,
    UCT_SCOPY_TX_PUT_ZCOPY,
    UCT_SCOPY_TX_FLUSH_COMP,
    UCT_SCOPY_TX_LAST
} uct_scopy_tx_op_t;


/**
 * TX operation executor
 *
 * @param [in]     tl_ep             Transport EP.
 * @param [in]     iov               The pointer to the array of UCT IOVs.
 * @param [in]     iov_cnt           The number of the elements in the array of UCT IOVs.
 * @param [in]     uct_iov_iter_p    The pointer to the UCT IOV iterator.
 * @param [in/out] length_p          Input: The maximal total length of the data that
 *                                   can be transferred in a single call. Output: The
 *                                   resulted length of the data that was transferred.
 * @param [in]     remote_addr       The address of the remote data buffer.
 * @param [in]     rkey              The remote memory key.
 * @param [in]     tx_op             TX operation identifier.
 *
 * @return UCS_OK if the operation was successfully completed, otherwise - error status.
 */
typedef ucs_status_t
(*uct_scopy_ep_tx_func_t)(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iov_cnt,
                          ucs_iov_iter_t *iov_iter_p, size_t *length_p,
                          uint64_t remote_addr, uct_rkey_t rkey,
                          uct_scopy_tx_op_t tx_op);


typedef struct uct_scopy_tx {
    ucs_arbiter_elem_t              arb_elem;           /* TX arbiter group element */
    uct_scopy_tx_op_t               op;                 /* TX operation identifier */
    uint64_t                        remote_addr;        /* The remote address */
    uct_rkey_t                      rkey;               /* User-passed UCT rkey */
    uct_completion_t                *comp;              /* The pointer to the user's passed completion */
    ucs_iov_iter_t                  iov_iter;           /* UCT IOVs iterator */
    size_t                          iov_cnt;            /* The number of the UCT IOVs */
    uct_iov_t                       iov[];              /* UCT IOVs */
} uct_scopy_tx_t;


typedef struct uct_scopy_ep {
    uct_base_ep_t                   super;
    ucs_arbiter_group_t             arb_group;          /* TX arbiter group */
} uct_scopy_ep_t;


UCS_CLASS_DECLARE(uct_scopy_ep_t, const uct_ep_params_t *);

ucs_status_t uct_scopy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                    size_t iov_cnt, uint64_t remote_addr,
                                    uct_rkey_t rkey, uct_completion_t *comp);

ucs_status_t uct_scopy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                    size_t iov_cnt, uint64_t remote_addr,
                                    uct_rkey_t rkey, uct_completion_t *comp);

ucs_arbiter_cb_result_t uct_scopy_ep_progress_tx(ucs_arbiter_t *arbiter,
                                                 ucs_arbiter_group_t *group,
                                                 ucs_arbiter_elem_t *elem,
                                                 void *arg);

ucs_status_t uct_scopy_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                uct_completion_t *comp);

#endif
