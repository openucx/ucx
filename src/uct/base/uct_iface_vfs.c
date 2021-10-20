/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_iface.h"
#include "uct_vfs_attr.h"
#include <uct/api/uct.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/string.h>
#include <ucs/vfs/base/vfs_obj.h>


static const uct_vfs_flag_info_t uct_iface_vfs_cap_infos[] = {
    {UCT_IFACE_FLAG_AM_SHORT, "am_short"},
    {UCT_IFACE_FLAG_AM_BCOPY, "am_bcopy"},
    {UCT_IFACE_FLAG_AM_ZCOPY, "am_zcopy"},
    {UCT_IFACE_FLAG_PENDING, "pending"},
    {UCT_IFACE_FLAG_PUT_SHORT, "put_short"},
    {UCT_IFACE_FLAG_PUT_BCOPY, "put_bcopy"},
    {UCT_IFACE_FLAG_PUT_ZCOPY, "put_zcopy"},
    {UCT_IFACE_FLAG_GET_SHORT, "get_short"},
    {UCT_IFACE_FLAG_GET_BCOPY, "get_bcopy"},
    {UCT_IFACE_FLAG_GET_ZCOPY, "get_zcopy"},
    {UCT_IFACE_FLAG_ATOMIC_CPU, "atomic_cpu"},
    {UCT_IFACE_FLAG_ATOMIC_DEVICE, "atomic_device"},
    {UCT_IFACE_FLAG_ERRHANDLE_SHORT_BUF, "errhandle_short_buf"},
    {UCT_IFACE_FLAG_ERRHANDLE_BCOPY_BUF, "errhandle_bcopy_buf"},
    {UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF, "errhandle_zcopy_buf"},
    {UCT_IFACE_FLAG_ERRHANDLE_AM_ID, "errhandle_am_id"},
    {UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM, "errhandle_remote_mem"},
    {UCT_IFACE_FLAG_ERRHANDLE_BCOPY_LEN, "errhandle_bcopy_len"},
    {UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE, "errhandle_peer_failure"},
    {UCT_IFACE_FLAG_EP_CHECK, "ep_check"},
    {UCT_IFACE_FLAG_CONNECT_TO_IFACE, "connect_to_iface"},
    {UCT_IFACE_FLAG_CONNECT_TO_EP, "connect_to_ep"},
    {UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR, "connect_to_sockaddr"},
    {UCT_IFACE_FLAG_AM_DUP, "am_dup"},
    {UCT_IFACE_FLAG_CB_SYNC, "cb_sync"},
    {UCT_IFACE_FLAG_CB_ASYNC, "cb_async"},
    {UCT_IFACE_FLAG_EP_KEEPALIVE, "ep_keepalive"},
    {UCT_IFACE_FLAG_TAG_EAGER_SHORT, "tag_eager_short"},
    {UCT_IFACE_FLAG_TAG_EAGER_BCOPY, "tag_eager_bcopy"},
    {UCT_IFACE_FLAG_TAG_EAGER_ZCOPY, "tag_eager_zcopy"},
    {UCT_IFACE_FLAG_TAG_RNDV_ZCOPY, "tag_rndv_zcopy"},
};

typedef struct {
    uint64_t   flag;
    const char *op_name;
    const char *limit_name;
    size_t     offset;
} uct_iface_vfs_cap_limit_info_t;

#define uct_iface_vfs_cap_limit_info(_flag, _op, _attr) \
    { \
        _flag, #_op, #_attr, ucs_offsetof(uct_iface_attr_t, cap._op._attr) \
    }

static const uct_iface_vfs_cap_limit_info_t uct_iface_vfs_cap_limit_infos[] = {
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_PUT_SHORT, put, max_short),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_PUT_BCOPY, put, max_bcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_PUT_ZCOPY, put, min_zcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_PUT_ZCOPY, put, max_zcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_PUT_ZCOPY, put,
                                 opt_zcopy_align),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_PUT_SHORT |
                                         UCT_IFACE_FLAG_PUT_BCOPY |
                                         UCT_IFACE_FLAG_PUT_ZCOPY,
                                 put, align_mtu),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_PUT_ZCOPY, put, max_iov),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_GET_SHORT, get, max_short),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_GET_BCOPY, get, max_bcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_GET_ZCOPY, get, min_zcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_GET_ZCOPY, get, max_zcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_GET_ZCOPY, get,
                                 opt_zcopy_align),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_GET_SHORT |
                                         UCT_IFACE_FLAG_GET_BCOPY |
                                         UCT_IFACE_FLAG_GET_ZCOPY,
                                 get, align_mtu),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_GET_ZCOPY, get, max_iov),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_SHORT, am, max_short),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_BCOPY, am, max_bcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_ZCOPY, am, min_zcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_ZCOPY, am, max_zcopy),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_ZCOPY, am, opt_zcopy_align),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_SHORT |
                                         UCT_IFACE_FLAG_AM_BCOPY |
                                         UCT_IFACE_FLAG_AM_ZCOPY,
                                 am, align_mtu),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_ZCOPY, am, max_hdr),
    uct_iface_vfs_cap_limit_info(UCT_IFACE_FLAG_AM_ZCOPY, am, max_iov),
};


static void uct_iface_vfs_show_cap_limit(void *obj, ucs_string_buffer_t *strb,
                                         void *arg_ptr, uint64_t arg_u64)
{
    uct_iface_h iface = obj;
    uct_iface_attr_t iface_attr;
    size_t attr;
    char buf[64];

    if (uct_iface_query(iface, &iface_attr) != UCS_OK) {
        ucs_string_buffer_appendf(strb, "<failed to query iface attributes>\n");
        return;
    }

    attr = *(size_t*)UCS_PTR_BYTE_OFFSET(&iface_attr, arg_u64);
    ucs_string_buffer_appendf(strb, "%s\n",
                              ucs_memunits_to_str(attr, buf, sizeof(buf)));
}

static void
uct_iface_vfs_init_cap_limits(uct_iface_h iface, uint64_t iface_cap_flags)
{
    size_t i;

    for (i = 0; i < ucs_static_array_size(uct_iface_vfs_cap_limit_infos); ++i) {
        if (iface_cap_flags & uct_iface_vfs_cap_limit_infos[i].flag) {
            ucs_vfs_obj_add_ro_file(iface, uct_iface_vfs_show_cap_limit, NULL,
                                    uct_iface_vfs_cap_limit_infos[i].offset,
                                    "capability/%s/%s",
                                    uct_iface_vfs_cap_limit_infos[i].op_name,
                                    uct_iface_vfs_cap_limit_infos[i].limit_name);
        }
    }
}

void uct_iface_vfs_refresh(void *obj)
{
    uct_base_iface_t *iface = obj;
    uct_iface_attr_t iface_attr;

    if (uct_iface_query(&iface->super, &iface_attr) == UCS_OK) {
        uct_vfs_init_flags(&iface->super, iface_attr.cap.flags,
                           uct_iface_vfs_cap_infos,
                           ucs_static_array_size(uct_iface_vfs_cap_infos));
        uct_iface_vfs_init_cap_limits(&iface->super, iface_attr.cap.flags);
    } else {
        ucs_debug("failed to query iface attributes");
    }

    iface->internal_ops->iface_vfs_refresh(&iface->super);
}
