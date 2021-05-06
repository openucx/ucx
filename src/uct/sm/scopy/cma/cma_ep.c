/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
#include <sys/uio.h>

#include "cma_ep.h"
#include <uct/base/uct_iov.inl>
#include <ucs/debug/log.h>
#include <ucs/sys/iovec.h>


typedef ssize_t (*uct_cma_ep_zcopy_fn_t)(pid_t, const struct iovec *,
                                         unsigned long, const struct iovec *,
                                         unsigned long, unsigned long);


const struct {
    uct_cma_ep_zcopy_fn_t fn;
    char                  *name;
} uct_cma_ep_fn[] = {
    [UCT_SCOPY_TX_GET_ZCOPY] = {
        .fn   = process_vm_readv,
        .name = "process_vm_readv"
    },
    [UCT_SCOPY_TX_PUT_ZCOPY] = {
        .fn   = process_vm_writev,
        .name = "process_vm_writev"
    }
};

static UCS_CLASS_INIT_FUNC(uct_cma_ep_t, const uct_ep_params_t *params)
{
    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_scopy_ep_t, params);

    self->remote_pid = *(const pid_t*)params->iface_addr &
                       ~UCT_CMA_IFACE_ADDR_FLAG_PID_NS;
    self->keepalive  = NULL;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cma_ep_t)
{
    ucs_free(self->keepalive);
}

UCS_CLASS_DEFINE(uct_cma_ep_t, uct_scopy_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cma_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cma_ep_t, uct_ep_t);

ucs_status_t uct_cma_ep_tx(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iov_cnt,
                           ucs_iov_iter_t *iov_iter, size_t *length_p,
                           uint64_t remote_addr, uct_rkey_t rkey,
                           uct_scopy_tx_op_t tx_op)
{
    uct_cma_ep_t *ep                   = ucs_derived_of(tl_ep, uct_cma_ep_t);
    uct_base_iface_t *iface            = ucs_derived_of(tl_ep->iface,
                                                        uct_base_iface_t);
    size_t local_iov_idx               = 0;
    size_t UCS_V_UNUSED remote_iov_idx = 0;
    size_t local_iov_cnt               = UCT_SM_MAX_IOV;
    size_t total_iov_length;
    struct iovec local_iov[UCT_SM_MAX_IOV], remote_iov;
    ucs_log_level_t log_lvl;
    ucs_status_t status;
    ssize_t ret;

    ucs_assert(*length_p != 0);

    total_iov_length = uct_iov_to_iovec(local_iov, &local_iov_cnt,
                                        iov, iov_cnt, *length_p, iov_iter);
    ucs_assert((total_iov_length <= *length_p) && (total_iov_length != 0) &&
               (local_iov_cnt > 0));

    remote_iov.iov_base = (void*)(uintptr_t)remote_addr;
    remote_iov.iov_len  = total_iov_length;

    ret = uct_cma_ep_fn[tx_op].fn(ep->remote_pid, &local_iov[local_iov_idx],
                                  local_iov_cnt - local_iov_idx, &remote_iov,
                                  1, 0);
    if (ucs_unlikely(ret < 0)) {
        status  = uct_iface_handle_ep_err(&iface->super, &ep->super.super.super,
                                          UCS_ERR_CONNECTION_RESET);
        log_lvl = uct_base_iface_failure_log_level(iface, status,
                                                   UCS_ERR_CONNECTION_RESET);

        ucs_log(log_lvl, "%s(pid=%d length=%zu) returned %zd: %m",
                uct_cma_ep_fn[tx_op].name, ep->remote_pid, remote_iov.iov_len,
                ret);

        return UCS_ERR_IO_ERROR;
    }

    ucs_assert(ret <= remote_iov.iov_len);

    *length_p = ret;
    return UCS_OK;
}

ucs_status_t uct_cma_ep_check(const uct_ep_h tl_ep, unsigned flags,
                              uct_completion_t *comp)
{
    uct_cma_ep_t *ep = ucs_derived_of(tl_ep, uct_cma_ep_t);

    return uct_ep_keepalive_check(tl_ep, &ep->keepalive, ep->remote_pid, flags,
                                  comp);
}
