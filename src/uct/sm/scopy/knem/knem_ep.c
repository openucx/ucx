/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <knem_io.h>

#include "knem_ep.h"
#include "knem_md.h"
#include <uct/base/uct_iov.inl>
#include <uct/sm/base/sm_iface.h>
#include <ucs/debug/log.h>


const char *uct_knem_ep_tx_op_str[] = {
    [UCT_SCOPY_TX_GET_ZCOPY] = "READ",
    [UCT_SCOPY_TX_PUT_ZCOPY] = "WRITE"
};


static UCS_CLASS_INIT_FUNC(uct_knem_ep_t, const uct_ep_params_t *params)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_scopy_ep_t, params);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_knem_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_knem_ep_t, uct_scopy_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_knem_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_knem_ep_t, uct_ep_t);

static UCS_F_ALWAYS_INLINE
void uct_knem_iovec_set_length(struct knem_cmd_param_iovec *iov, size_t length)
{
    iov->len = length;
}

static UCS_F_ALWAYS_INLINE
void uct_knem_iovec_set_buffer(struct knem_cmd_param_iovec *iov, void *buffer)
{
    iov->base = (uintptr_t)buffer;
}

ucs_status_t uct_knem_ep_tx(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iov_cnt,
                            ucs_iov_iter_t *iov_iter, size_t *length_p,
                            uint64_t remote_addr, uct_rkey_t rkey,
                            uct_scopy_tx_op_t tx_op)
{
    uct_knem_iface_t *knem_iface = ucs_derived_of(tl_ep->iface,
                                                  uct_knem_iface_t);
    int knem_fd                  = knem_iface->knem_md->knem_fd;
    uct_knem_key_t *key          = (uct_knem_key_t*)rkey;
    size_t local_iov_cnt         = UCT_SM_MAX_IOV;
    struct knem_cmd_param_iovec local_iov[UCT_SM_MAX_IOV];
    size_t UCS_V_UNUSED total_iov_length;
    struct knem_cmd_inline_copy icopy;
    int i, ret;

    ucs_assert(*length_p != 0);

    total_iov_length = ucs_iov_converter(local_iov, &local_iov_cnt,
                                         uct_knem_iovec_set_buffer, uct_knem_iovec_set_length,
                                         iov, iov_cnt,
                                         uct_iov_get_buffer, uct_iov_get_length,
                                         *length_p, iov_iter);
    ucs_assert((total_iov_length <= *length_p) && (total_iov_length != 0) &&
               (local_iov_cnt > 0));

    icopy.local_iovec_array = (uintptr_t)local_iov;
    icopy.local_iovec_nr    = local_iov_cnt;
    icopy.remote_cookie     = key->cookie;
    icopy.current_status    = 0;
    ucs_assert(remote_addr >= key->address);
    icopy.remote_offset     = remote_addr - key->address;
    /* This value is used to set `knem_cmd_inline_copy::write` field */
    UCS_STATIC_ASSERT(UCT_SCOPY_TX_PUT_ZCOPY == 1);
    icopy.write             = tx_op;
    /* TBD: add check and support for KNEM_FLAG_DMA */
    icopy.flags             = 0;

    ucs_assert(knem_fd > -1);
    ret = ioctl(knem_fd, KNEM_CMD_INLINE_COPY, &icopy);
    if (ucs_unlikely((ret < 0) ||
                     (icopy.current_status != KNEM_STATUS_SUCCESS))) {
        ucs_error("KNEM inline copy \"%s\" failed, ioctl() return value - %d, "
                  "copy status - %d: %m",
                  uct_knem_ep_tx_op_str[tx_op], ret, icopy.current_status);
        return UCS_ERR_IO_ERROR;
    }

    if (tx_op == UCT_SCOPY_TX_GET_ZCOPY) {
        for (i = 0; i < local_iov_cnt; ++i) {
            VALGRIND_MAKE_MEM_DEFINED(local_iov[i].base, local_iov[i].len);
        }
    }
    *length_p = total_iov_length;
    return UCS_OK;
}
