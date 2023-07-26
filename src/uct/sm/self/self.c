/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "self.h"

#include <uct/base/uct_iov.inl>
#include <uct/sm/base/sm_ep.h>
#include <uct/sm/base/sm_iface.h>
#include <uct/sm/base/sm_md.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>
#include <ucs/arch/cpu.h>
#include <uct/api/v2/uct_v2.h>
#include "self.h"


#define UCT_SELF_NAME "self"

#define UCT_SELF_IFACE_SEND_BUFFER_GET(_iface) \
    ({ /* use buffers from mpool to avoid buffer re-usage */ \
       /* till operation completes */ \
        void *ptr = ucs_mpool_get_inline(&(_iface)->msg_mp); \
        if (ucs_unlikely(ptr == NULL)) { \
                return UCS_ERR_NO_MEMORY; \
        } \
        ptr; \
    })


/* Forward declarations */
static uct_iface_ops_t uct_self_iface_ops;
static uct_component_t uct_self_component;


static ucs_config_field_t uct_self_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_self_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"SEG_SIZE", "8k",
     "Size of copy-out buffer",
     ucs_offsetof(uct_self_iface_config_t, seg_size), UCS_CONFIG_TYPE_MEMUNITS},

    {NULL}
};

static ucs_config_field_t uct_self_md_config_table[] = {
    {"", "", NULL, ucs_offsetof(uct_self_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"NUM_DEVICES", "1", "Number of \"self\" devices to create",
     ucs_offsetof(uct_self_md_config_t, num_devices), UCS_CONFIG_TYPE_INT},

    {NULL}
};


static ucs_status_t uct_self_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *attr)
{
    uct_self_iface_t *iface = ucs_derived_of(tl_iface, uct_self_iface_t);

    ucs_trace_func("iface=%p", iface);

    uct_base_iface_query(&iface->super, attr);

    attr->iface_addr_len         = sizeof(uct_self_iface_addr_t);
    attr->device_addr_len        = 0;
    attr->ep_addr_len            = 0;
    attr->max_conn_priv          = 0;
    attr->cap.flags              = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                   UCT_IFACE_FLAG_AM_SHORT         |
                                   UCT_IFACE_FLAG_AM_BCOPY         |
                                   UCT_IFACE_FLAG_PUT_SHORT        |
                                   UCT_IFACE_FLAG_PUT_BCOPY        |
                                   UCT_IFACE_FLAG_GET_BCOPY        |
                                   UCT_IFACE_FLAG_ATOMIC_CPU       |
                                   UCT_IFACE_FLAG_PENDING          |
                                   UCT_IFACE_FLAG_CB_SYNC          |
                                   UCT_IFACE_FLAG_EP_CHECK;

    attr->cap.atomic32.op_flags   =
    attr->cap.atomic64.op_flags   = UCS_BIT(UCT_ATOMIC_OP_ADD)     |
                                    UCS_BIT(UCT_ATOMIC_OP_AND)     |
                                    UCS_BIT(UCT_ATOMIC_OP_OR)      |
                                    UCS_BIT(UCT_ATOMIC_OP_XOR);
    attr->cap.atomic32.fop_flags  =
    attr->cap.atomic64.fop_flags  = UCS_BIT(UCT_ATOMIC_OP_ADD)     |
                                    UCS_BIT(UCT_ATOMIC_OP_AND)     |
                                    UCS_BIT(UCT_ATOMIC_OP_OR)      |
                                    UCS_BIT(UCT_ATOMIC_OP_XOR)     |
                                    UCS_BIT(UCT_ATOMIC_OP_SWAP)    |
                                    UCS_BIT(UCT_ATOMIC_OP_CSWAP);

    attr->cap.put.max_short       = UINT_MAX;
    attr->cap.put.max_bcopy       = SIZE_MAX;
    attr->cap.put.min_zcopy       = 0;
    attr->cap.put.max_zcopy       = 0;
    attr->cap.put.opt_zcopy_align = 1;
    attr->cap.put.align_mtu       = attr->cap.put.opt_zcopy_align;
    attr->cap.put.max_iov         = 1;

    attr->cap.get.max_bcopy       = SIZE_MAX;
    attr->cap.get.min_zcopy       = 0;
    attr->cap.get.max_zcopy       = 0;
    attr->cap.get.opt_zcopy_align = 1;
    attr->cap.get.align_mtu       = attr->cap.get.opt_zcopy_align;
    attr->cap.get.max_iov         = 1;

    attr->cap.am.max_short        = iface->send_size;
    attr->cap.am.max_bcopy        = iface->send_size;
    attr->cap.am.min_zcopy        = 0;
    attr->cap.am.max_zcopy        = 0;
    attr->cap.am.opt_zcopy_align  = 1;
    attr->cap.am.align_mtu        = attr->cap.am.opt_zcopy_align;
    attr->cap.am.max_hdr          = 0;
    attr->cap.am.max_iov          = SIZE_MAX;

    attr->latency                 = UCS_LINEAR_FUNC_ZERO;
    attr->bandwidth.dedicated     = 19360 * UCS_MBYTE;
    attr->bandwidth.shared        = 0;
    attr->overhead                = 10e-9;
    attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t uct_self_iface_get_address(uct_iface_h tl_iface,
                                               uct_iface_addr_t *addr)
{
    const uct_self_iface_t *iface = ucs_derived_of(tl_iface, uct_self_iface_t);

    *(uct_self_iface_addr_t*)addr = iface->id;
    return UCS_OK;
}

static int uct_self_iface_is_reachable(const uct_iface_h tl_iface,
                                       const uct_device_addr_t *dev_addr,
                                       const uct_iface_addr_t *iface_addr)
{
    const uct_self_iface_t     *iface = ucs_derived_of(tl_iface, uct_self_iface_t);
    const uct_self_iface_addr_t *addr = (const uct_self_iface_addr_t*)iface_addr;

    return (addr != NULL) && (iface->id == *addr);
}

static void uct_self_iface_sendrecv_am(uct_self_iface_t *iface, uint8_t am_id,
                                       void *buffer, size_t length, const char *title)
{
    ucs_status_t UCS_V_UNUSED status;

    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, am_id,
                       buffer, length, "TX: AM_%s", title);
    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_RECV, am_id,
                       buffer, length, "RX: AM_%s", title);

    status = uct_iface_invoke_am(&iface->super, am_id, buffer,
                                 length, 0);
    ucs_assert(status == UCS_OK);
    ucs_mpool_put_inline(buffer);
}

static ucs_mpool_ops_t uct_self_iface_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

static UCS_CLASS_INIT_FUNC(uct_self_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_self_iface_config_t *config = ucs_derived_of(tl_config,
                                                     uct_self_iface_config_t);
    size_t align_offset, alignment;
    ucs_status_t status;
    ucs_mpool_params_t mp_params;

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");
    if (!(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE)) {
        ucs_error("Self transport supports only UCT_IFACE_OPEN_MODE_DEVICE");
        return UCS_ERR_UNSUPPORTED;
    }

    if (ucs_derived_of(worker, uct_priv_worker_t)->thread_mode == UCS_THREAD_MODE_MULTI) {
        ucs_error("Self transport does not support multi-threaded worker");
        return UCS_ERR_INVALID_PARAM;
    }

    UCS_CLASS_CALL_SUPER_INIT(
            uct_base_iface_t, &uct_self_iface_ops,
            &uct_base_iface_internal_ops, md, worker, params,
            tl_config UCS_STATS_ARG(
                    (params->field_mask & UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                            params->stats_root :
                            NULL) UCS_STATS_ARG(UCT_SELF_NAME));

    self->id          = ucs_generate_uuid((uintptr_t)self);
    self->send_size   = config->seg_size;

    status = uct_iface_param_am_alignment(params, self->send_size, 0, 0,
                                          &alignment, &align_offset);
    if (status != UCS_OK) {
        return status;
    }

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = self->send_size;
    mp_params.align_offset    = align_offset;
    mp_params.alignment       = alignment;
    mp_params.elems_per_chunk = 2; /* 2 elements are enough for most of communications */
    mp_params.ops             = &uct_self_iface_mpool_ops;
    mp_params.name            = "self_msg_desc";
    status = ucs_mpool_init(&mp_params, &self->msg_mp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    ucs_debug("created self iface id 0x%"PRIx64" send_size %zu", self->id,
              self->send_size);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_self_iface_t)
{
    ucs_mpool_cleanup(&self->msg_mp, 1);
}

UCS_CLASS_DEFINE(uct_self_iface_t, uct_base_iface_t);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_self_iface_t, uct_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_self_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static ucs_status_t
uct_self_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                          unsigned *num_tl_devices_p)
{
    uct_self_md_t *self_md = ucs_derived_of(md, uct_self_md_t);
    int i;
    uct_tl_device_resource_t *devices;

    devices = ucs_calloc(self_md->num_devices, sizeof(*devices),
                         "device resource");
    if (NULL == devices) {
        ucs_error("failed to allocate device resource");
        return UCS_ERR_NO_MEMORY;
    }

    for (i = 0; i < self_md->num_devices; i++) {
        if (self_md->num_devices > 1) {
            ucs_snprintf_zero(devices[i].name, sizeof(devices->name), "%s%d",
                              UCT_SM_DEVICE_NAME, i);
        } else {
            ucs_strncpy_zero(devices[i].name, UCT_SM_DEVICE_NAME,
                             sizeof(devices->name));
        }
        devices[i].type       = UCT_DEVICE_TYPE_SELF;
        devices[i].sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    *tl_devices_p     = devices;
    *num_tl_devices_p = self_md->num_devices;
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_self_ep_t, const uct_ep_params_t *params)
{
    uct_self_iface_t *iface = ucs_derived_of(params->iface, uct_self_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_self_ep_t)
{
}

UCS_CLASS_DEFINE(uct_self_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_self_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_self_ep_t, uct_ep_t);


ucs_status_t uct_self_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length)
{
    uct_self_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_self_iface_t);
    uct_self_ep_t UCS_V_UNUSED *ep = ucs_derived_of(tl_ep, uct_self_ep_t);
    size_t total_length;
    void *send_buffer;

    UCT_CHECK_AM_ID(id);

    total_length = length + sizeof(header);
    UCT_CHECK_LENGTH(total_length, 0, iface->send_size, "am_short");

    send_buffer = UCT_SELF_IFACE_SEND_BUFFER_GET(iface);
    uct_am_short_fill_data(send_buffer, header, payload, length);

    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, total_length);
    uct_self_iface_sendrecv_am(iface, id, send_buffer, total_length, "SHORT");
    return UCS_OK;
}

ucs_status_t uct_self_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                      const uct_iov_t *iov, size_t iovcnt)
{
    uct_self_iface_t *iface        = ucs_derived_of(tl_ep->iface,
                                                    uct_self_iface_t);
    uct_self_ep_t UCS_V_UNUSED *ep = ucs_derived_of(tl_ep, uct_self_ep_t);
    void *send_buffer;
    ucs_iov_iter_t iov_iter;
    size_t length;

    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt), 0, iface->send_size,
                     "am_short_iov");

    ucs_iov_iter_init(&iov_iter);
    send_buffer = UCT_SELF_IFACE_SEND_BUFFER_GET(iface);
    length      = uct_iov_to_buffer(iov, iovcnt, &iov_iter, send_buffer,
                                    SIZE_MAX);

    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, length);
    uct_self_iface_sendrecv_am(iface, id, send_buffer, length, "SHORT_IOV");

    return UCS_OK;
}

ssize_t uct_self_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                             uct_pack_callback_t pack_cb, void *arg,
                             unsigned flags)
{
    uct_self_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_self_iface_t);
    uct_self_ep_t UCS_V_UNUSED *ep = ucs_derived_of(tl_ep, uct_self_ep_t);
    size_t length;
    void *send_buffer;

    UCT_CHECK_AM_ID(id);

    send_buffer = UCT_SELF_IFACE_SEND_BUFFER_GET(iface);
    length = pack_cb(send_buffer, arg);

    UCT_CHECK_LENGTH(length, 0, iface->send_size, "am_bcopy");
    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);

    uct_self_iface_sendrecv_am(iface, id, send_buffer, length, "BCOPY");
    return length;
}

static uct_iface_ops_t uct_self_iface_ops = {
    .ep_put_short             = uct_sm_ep_put_short,
    .ep_put_bcopy             = uct_sm_ep_put_bcopy,
    .ep_get_bcopy             = uct_sm_ep_get_bcopy,
    .ep_am_short              = uct_self_ep_am_short,
    .ep_am_short_iov          = uct_self_ep_am_short_iov,
    .ep_am_bcopy              = uct_self_ep_am_bcopy,
    .ep_atomic_cswap64        = uct_sm_ep_atomic_cswap64,
    .ep_atomic64_post         = uct_sm_ep_atomic64_post,
    .ep_atomic64_fetch        = uct_sm_ep_atomic64_fetch,
    .ep_atomic_cswap32        = uct_sm_ep_atomic_cswap32,
    .ep_atomic32_post         = uct_sm_ep_atomic32_post,
    .ep_atomic32_fetch        = uct_sm_ep_atomic32_fetch,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = ucs_empty_function_return_success,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_self_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_self_ep_t),
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_self_iface_t),
    .iface_query              = uct_self_iface_query,
    .iface_get_device_address = ucs_empty_function_return_success,
    .iface_get_address        = uct_self_iface_get_address,
    .iface_is_reachable       = uct_self_iface_is_reachable
};

static ucs_status_t uct_self_md_query(uct_md_h md, uct_md_attr_v2_t *attr)
{
    /* Dummy memory registration provided. No real memory handling exists */
    attr->flags                  = UCT_MD_FLAG_REG |
                                   UCT_MD_FLAG_NEED_RKEY; /* TODO ignore rkey in rma/amo ops */
    attr->reg_mem_types          = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->reg_nonblock_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->cache_mem_types        = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->alloc_mem_types        = 0;
    attr->detect_mem_types       = 0;
    attr->access_mem_types       = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->dmabuf_mem_types       = 0;
    attr->max_alloc              = 0;
    attr->max_reg                = ULONG_MAX;
    attr->rkey_packed_size       = 0;
    attr->reg_cost               = UCS_LINEAR_FUNC_ZERO;
    memset(&attr->local_cpus, 0xff, sizeof(attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_self_md_open(uct_component_t *component, const char *md_name,
                                     const uct_md_config_t *config, uct_md_h *md_p)
{
    uct_self_md_config_t *md_config = ucs_derived_of(config,
                                                     uct_self_md_config_t);
    static uct_md_ops_t md_ops = {
        .close              = ucs_empty_function,
        .query              = uct_self_md_query,
        .mkey_pack          = ucs_empty_function_return_success,
        .mem_reg            = uct_md_dummy_mem_reg,
        .mem_dereg          = uct_md_dummy_mem_dereg,
        .mem_attach         = ucs_empty_function_return_unsupported,
        .detect_memory_type = ucs_empty_function_return_unsupported
    };

    static uct_self_md_t md;

    md.super.ops       = &md_ops;
    md.super.component = &uct_self_component;
    md.num_devices     = md_config->num_devices;

    *md_p = &md.super;
    return UCS_OK;
}

static ucs_status_t uct_self_md_rkey_unpack(uct_component_t *component,
                                            const void *rkey_buffer, uct_rkey_t *rkey_p,
                                            void **handle_p)
{
    /**
     * Pseudo stub function for the key unpacking
     * Need rkey == 0 due to work with same process to reuse uct_base_[put|get|atomic]*
     */
    *rkey_p   = 0;
    *handle_p = NULL;
    return UCS_OK;
}

static uct_component_t uct_self_component = {
    .query_md_resources = uct_md_query_single_md_resource,
    .md_open            = uct_self_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_self_md_rkey_unpack,
    .rkey_ptr           = uct_sm_rkey_ptr,
    .rkey_release       = ucs_empty_function_return_success,
    .name               = UCT_SELF_NAME,
    .md_config          = {
        .name           = "Self memory domain",
        .prefix         = "SELF_",
        .table          = uct_self_md_config_table,
        .size           = sizeof(uct_self_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_self_component),
    .flags              = UCT_COMPONENT_FLAG_RKEY_PTR,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};

UCT_TL_DEFINE_ENTRY(&uct_self_component, self, uct_self_query_tl_devices,
                    uct_self_iface_t, "SELF_", uct_self_iface_config_table,
                    uct_self_iface_config_t);

UCT_SINGLE_TL_INIT(&uct_self_component, self,,,)
