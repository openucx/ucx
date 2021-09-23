/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "scopy_iface.h"
#include "scopy_ep.h"

#include <ucs/arch/cpu.h>
#include <ucs/sys/string.h>

#include <uct/sm/base/sm_iface.h>


ucs_config_field_t uct_scopy_iface_config_table[] = {
    {"SM_", "", NULL,
     ucs_offsetof(uct_scopy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_sm_iface_config_table)},

    {"MAX_IOV", "16",
     "Maximum IOV count that can contain user-defined payload in a single\n"
     "call to GET/PUT Zcopy operation",
     ucs_offsetof(uct_scopy_iface_config_t, max_iov), UCS_CONFIG_TYPE_ULONG},

    {"SEG_SIZE", "512k",
     "Segment size that is used to perform data transfer when doing progress\n"
     "of GET/PUT Zcopy operations",
     ucs_offsetof(uct_scopy_iface_config_t, seg_size), UCS_CONFIG_TYPE_MEMUNITS},

    /* TX_QUOTA=1 is used by default in order to make iface progress more
     * lightweight and not be blocked for a long time (CMA/KNEM write/read
     * operations are blocking). The blocking iface progress for a long time
     * is harmful for the many-to-one (GET operation) and one-to-many (PUT
     * operation) patterns. */
    {"TX_QUOTA", "1",
     "How many TX segments can be dispatched during iface progress",
     ucs_offsetof(uct_scopy_iface_config_t, tx_quota), UCS_CONFIG_TYPE_UINT},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("TX_", -1, 8, "send",
                                  ucs_offsetof(uct_scopy_iface_config_t, tx_mpool), ""),

    {NULL}
};

static ucs_mpool_ops_t uct_scopy_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

void uct_scopy_iface_query(uct_scopy_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    uct_base_iface_query(&iface->super.super, iface_attr);

    /* default values for all shared memory transports */
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = iface->config.max_iov;

    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = iface->config.max_iov;

    iface_attr->device_addr_len         = uct_sm_iface_get_device_addr_len();
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING   |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->cap.event_flags         = UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                          UCT_IFACE_FLAG_EVENT_RECV      |
                                          UCT_IFACE_FLAG_EVENT_ASYNC_CB;
    iface_attr->latency                 = ucs_linear_func_make(80e-9, 0); /* 80 ns */
    iface_attr->overhead                = (ucs_arch_get_cpu_vendor() ==
                                           UCS_CPU_VENDOR_FUJITSU_ARM) ?
                                          6e-6 : 2e-6;
}

UCS_CLASS_INIT_FUNC(uct_scopy_iface_t, uct_iface_ops_t *ops,
                    uct_scopy_iface_ops_t *scopy_ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config)
{
    uct_scopy_iface_config_t *config = ucs_derived_of(tl_config,
                                                      uct_scopy_iface_config_t);
    size_t elem_size;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_sm_iface_t, ops, &scopy_ops->super, md,
                              worker, params, tl_config);

    self->tx              = scopy_ops->ep_tx;
    self->config.max_iov  = ucs_min(config->max_iov, ucs_iov_get_max());
    self->config.seg_size = config->seg_size;
    self->config.tx_quota = config->tx_quota;

    elem_size             = sizeof(uct_scopy_tx_t) +
                            self->config.max_iov * sizeof(uct_iov_t);

    ucs_arbiter_init(&self->arbiter);

    status = ucs_mpool_init(&self->tx_mpool, 0, elem_size,
                            0, UCS_SYS_CACHE_LINE_SIZE,
                            config->tx_mpool.bufs_grow,
                            config->tx_mpool.max_bufs,
                            &uct_scopy_mpool_ops,
                            "uct_scopy_iface_tx_mp");

    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_scopy_iface_t)
{
    uct_worker_progress_unregister_safe(&self->super.super.worker->super,
                                        &self->super.super.prog.id);
    ucs_mpool_cleanup(&self->tx_mpool, 1);
    ucs_arbiter_cleanup(&self->arbiter);
}

UCS_CLASS_DEFINE(uct_scopy_iface_t, uct_sm_iface_t);

unsigned uct_scopy_iface_progress(uct_iface_h tl_iface)
{
    uct_scopy_iface_t *iface = ucs_derived_of(tl_iface, uct_scopy_iface_t);
    unsigned count           = 0;

    ucs_arbiter_dispatch(&iface->arbiter, 1, uct_scopy_ep_progress_tx, &count);

    if (ucs_unlikely(ucs_arbiter_is_empty(&iface->arbiter))) {
        uct_worker_progress_unregister_safe(&iface->super.super.worker->super,
                                            &iface->super.super.prog.id);
    }

    return count;
}

ucs_status_t uct_scopy_iface_event_arm(uct_iface_h tl_iface, unsigned events)
{
    uct_scopy_iface_t *iface = ucs_derived_of(tl_iface, uct_scopy_iface_t);

    if ((events & UCT_EVENT_SEND_COMP) &&
        !ucs_arbiter_is_empty(&iface->arbiter)) {
        /* cannot go to sleep, need to progress pending operations */
        return UCS_ERR_BUSY;
    }

    return UCS_OK;
}

ucs_status_t uct_scopy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                   uct_completion_t *comp)
{
    uct_scopy_iface_t *iface = ucs_derived_of(tl_iface, uct_scopy_iface_t);

    if (ucs_unlikely(comp != NULL)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!ucs_arbiter_is_empty(&iface->arbiter)) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super);
        return UCS_INPROGRESS;
    }

    UCT_TL_IFACE_STAT_FLUSH(&iface->super.super);
    return UCS_OK;
}
