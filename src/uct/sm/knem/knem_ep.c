/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <knem_io.h>

#include "knem_ep.h"
#include "knem_pd.h"
#include <ucs/debug/log.h>

static UCS_CLASS_INIT_FUNC(uct_knem_ep_t, uct_iface_t *tl_iface,
                           const struct sockaddr *addr)
{
    uct_knem_iface_t *iface = ucs_derived_of(tl_iface, uct_knem_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_knem_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_knem_ep_t, uct_base_ep_t)
    UCS_CLASS_DEFINE_NEW_FUNC(uct_knem_ep_t, uct_ep_t, uct_iface_t*,
                              const struct sockaddr *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_knem_ep_t, uct_ep_t);


#define uct_knem_trace_data(_remote_addr, _rkey, _fmt, ...) \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                   (_rkey))

#define UCT_KNEM_ZERO_LENGTH_POST(len)              \
    if (0 == len) {                                     \
        ucs_trace_data("Zero length request: skip it"); \
        return UCS_OK;                                  \
    }

static inline ucs_status_t uct_knem_rma(uct_ep_h tl_ep, const void *buffer,
                                        size_t length,  uint64_t remote_addr,
                                        uct_knem_key_t *key, int write)
{
    struct knem_cmd_inline_copy icopy;
    struct knem_cmd_param_iovec knem_iov[1];
    uct_knem_iface_t *knem_iface = ucs_derived_of(tl_ep->iface, uct_knem_iface_t);
    int knem_fd = knem_iface->knem_pd->knem_fd;
    int rc;

    UCT_KNEM_ZERO_LENGTH_POST(length);

    knem_iov[0].base = (uintptr_t)buffer;
    knem_iov[0].len = length;

    icopy.local_iovec_array = (uintptr_t) &knem_iov[0];
    icopy.local_iovec_nr = 1;
    icopy.remote_cookie = key->cookie;
    ucs_assert(remote_addr >= key->address);
    icopy.remote_offset = remote_addr - key->address;

    icopy.write = write; /* if 0 then, READ from the remote region into my local segments
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
                        write?"PUT_ZCOPY":"GET_ZCOPY",
                        length);
    return UCS_OK;
}

ucs_status_t uct_knem_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_knem_key_t *key = (uct_knem_key_t *)rkey;
    ucs_status_t status;
    
    status = uct_knem_rma(tl_ep, buffer, length, remote_addr, key, 1);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY, length);
    return status;
}

ucs_status_t uct_knem_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_knem_key_t *key = (uct_knem_key_t *)rkey;
    ucs_status_t status;

    status = uct_knem_rma(tl_ep, buffer, length, remote_addr, key, 0);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY, length);
    return status;
}
