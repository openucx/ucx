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
#include <ucs/debug/log.h>
#include <ucs/sys/iovec.h>

typedef ssize_t (*uct_cma_ep_zcopy_fn_t)(pid_t, const struct iovec *,
                                         unsigned long, const struct iovec *,
                                         unsigned long, unsigned long);

static UCS_CLASS_INIT_FUNC(uct_cma_ep_t, const uct_ep_params_t *params)
{
    uct_cma_iface_t *iface = ucs_derived_of(params->iface, uct_cma_iface_t);

    UCT_CHECK_PARAM(params->field_mask & UCT_EP_PARAM_FIELD_IFACE_ADDR,
                    "UCT_EP_PARAM_FIELD_IFACE_ADDR and UCT_EP_PARAM_FIELD_DEV_ADDR are not defined");

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);
    self->remote_pid = *(const pid_t*)params->iface_addr &
                       ~UCT_CMA_IFACE_ADDR_FLAG_PID_NS;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cma_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_cma_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cma_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cma_ep_t, uct_ep_t);


#define uct_cma_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

static UCS_F_ALWAYS_INLINE
ucs_status_t uct_cma_ep_do_zcopy(uct_cma_ep_t *ep, struct iovec *local_iov,
                                 size_t local_iov_cnt, struct iovec *remote_iov,
                                 uct_cma_ep_zcopy_fn_t fn_p, const char *fn_name)
{
    size_t local_iov_idx               = 0;
    size_t UCS_V_UNUSED remove_iov_idx = 0;
    ssize_t ret;

    do {
        ret = fn_p(ep->remote_pid, &local_iov[local_iov_idx],
                   local_iov_cnt - local_iov_idx, remote_iov, 1, 0);
        if (ucs_unlikely(ret < 0)) {
            ucs_error("%s(pid=%d length=%zu) returned %zd: %m",
                      fn_name, ep->remote_pid, remote_iov->iov_len, ret);
            return UCS_ERR_IO_ERROR;
        }

        ucs_assert(ret <= remote_iov->iov_len);
        ucs_iov_advance(local_iov, local_iov_cnt, &local_iov_idx, ret);
        ucs_iov_advance(remote_iov, 1, &remove_iov_idx, ret);
    } while (remote_iov->iov_len);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t uct_cma_ep_common_zcopy(uct_ep_h tl_ep,
                                     const uct_iov_t *iov,
                                     size_t iovcnt,
                                     uint64_t remote_addr,
                                     uct_completion_t *comp,
                                     ssize_t (*fn_p)(pid_t,
                                                     const struct iovec *,
                                                     unsigned long,
                                                     const struct iovec *,
                                                     unsigned long,
                                                     unsigned long),
                                     const char *fn_name)
{
    uct_cma_ep_t *ep = ucs_derived_of(tl_ep, uct_cma_ep_t);
    size_t iov_idx   = 0;
    ucs_status_t status;
    size_t local_iov_cnt;
    size_t length;
    size_t cur_iov_cnt;
    struct iovec local_iov[UCT_SM_MAX_IOV];
    struct iovec remote_iov;

    remote_iov.iov_base = (void*)remote_addr;

    while (iov_idx < iovcnt) {
        cur_iov_cnt   = ucs_min(iovcnt - iov_idx, UCT_SM_MAX_IOV);
        local_iov_cnt = uct_iovec_fill_iov(local_iov, &iov[iov_idx],
                                           cur_iov_cnt, &length);
        ucs_assert(local_iov_cnt <= cur_iov_cnt);

        iov_idx += cur_iov_cnt;
        ucs_assert(iov_idx <= iovcnt);

        if (!length) {
            continue; /* Nothing to deliver */
        }

        remote_iov.iov_len = length;

        status = uct_cma_ep_do_zcopy(ep, local_iov, local_iov_cnt,
                                     &remote_iov, fn_p, fn_name);
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t uct_cma_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                  uint64_t remote_addr, uct_rkey_t rkey,
                                  uct_completion_t *comp)
{
    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_cma_ep_put_zcopy");

    ucs_status_t ret = uct_cma_ep_common_zcopy(tl_ep,
                                               iov,
                                               iovcnt,
                                               remote_addr,
                                               comp,
                                               process_vm_writev,
                                               "process_vm_writev");

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cma_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                       uct_iov_total_length(iov, iovcnt));
    return ret;
}

ucs_status_t uct_cma_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                  uint64_t remote_addr, uct_rkey_t rkey,
                                  uct_completion_t *comp)
{
    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_cma_ep_get_zcopy");

    ucs_status_t ret = uct_cma_ep_common_zcopy(tl_ep,
                                               iov,
                                               iovcnt,
                                               remote_addr,
                                               comp,
                                               process_vm_readv,
                                               "process_vm_readv");

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cma_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                       uct_iov_total_length(iov, iovcnt));
    return ret;
}
