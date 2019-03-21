/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <knem_io.h>

#include "knem_ep.h"
#include "knem_md.h"
#include <uct/sm/base/sm_iface.h>
#include <ucs/debug/log.h>

static UCS_CLASS_INIT_FUNC(uct_knem_ep_t, const uct_ep_params_t *params)
{
    uct_knem_iface_t *iface = ucs_derived_of(params->iface, uct_knem_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_knem_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_knem_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_knem_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_knem_ep_t, uct_ep_t);


#define uct_knem_trace_data(_remote_addr, _rkey, _fmt, ...) \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                   (_rkey))

#define UCT_KNEM_ZERO_LENGTH_POST(len)              \
    if (0 == len) {                                     \
        ucs_trace_data("Zero length request: skip it"); \
        return UCS_OK;                                  \
    }

static inline ucs_status_t uct_knem_rma(uct_ep_h tl_ep, const uct_iov_t *iov,
                                        size_t iovcnt, uint64_t remote_addr,
                                        uct_knem_key_t *key, int ucs_write)
{
    struct knem_cmd_inline_copy icopy;
    struct knem_cmd_param_iovec knem_iov[UCT_SM_MAX_IOV];
    uct_knem_iface_t *knem_iface = ucs_derived_of(tl_ep->iface, uct_knem_iface_t);
    int knem_fd = knem_iface->knem_md->knem_fd;
    int rc;
    size_t iov_it;
    size_t knem_iov_it = 0;

    for (iov_it = 0; iov_it < ucs_min(UCT_SM_MAX_IOV, iovcnt); ++iov_it) {
        knem_iov[knem_iov_it].base = (uintptr_t)iov[iov_it].buffer;
        knem_iov[knem_iov_it].len = uct_iov_get_length(iov + iov_it);
        if (knem_iov[knem_iov_it].len) {
            ++knem_iov_it;
        } else {
            continue; /* Skip zero length buffers */
        }
    }

    UCT_KNEM_ZERO_LENGTH_POST(knem_iov_it);

    icopy.local_iovec_array = (uintptr_t) knem_iov;
    icopy.local_iovec_nr = knem_iov_it;
    icopy.remote_cookie = key->cookie;
    ucs_assert(remote_addr >= key->address);
    icopy.remote_offset = remote_addr - key->address;

    icopy.write = ucs_write; /* if 0 then, READ from the remote region into my local segments
                          * if 1 then, WRITE to the remote region from my local segment */
    icopy.flags = 0;     /* TBD: add check and support for KNEM_FLAG_DMA */
    icopy.current_status = 0;
    icopy.async_status_index = 0;
    icopy.pad = 0;

    ucs_assert(knem_fd > -1);
    rc = ioctl(knem_fd, KNEM_CMD_INLINE_COPY, &icopy);
    if (rc < 0) {
        ucs_error("KNEM inline copy failed, err = %d %m", rc);
        return UCS_ERR_IO_ERROR;
    }

    uct_knem_trace_data(remote_addr, (uintptr_t)key, "%s [length %zu]",
                        ucs_write?"PUT_ZCOPY":"GET_ZCOPY",
                        uct_iov_total_length(iov, iovcnt));
    return UCS_OK;
}

ucs_status_t uct_knem_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp)
{
    uct_knem_key_t *key = (uct_knem_key_t *)rkey;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_knem_ep_put_zcopy");

    status = uct_knem_rma(tl_ep, iov, iovcnt, remote_addr, key, 1);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, ucs_derived_of(tl_ep, uct_base_ep_t),
                                 PUT, ZCOPY, uct_iov_total_length(iov, iovcnt));
    return status;
}

ucs_status_t uct_knem_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp)
{
    uct_knem_key_t *key = (uct_knem_key_t *)rkey;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_knem_ep_get_zcopy");

    status = uct_knem_rma(tl_ep, iov, iovcnt, remote_addr, key, 0);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, ucs_derived_of(tl_ep, uct_base_ep_t),
                                 GET, ZCOPY, uct_iov_total_length(iov, iovcnt));
    return status;
}
