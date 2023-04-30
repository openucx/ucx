/**
 * Copyright (C) UT-Battelle, LLC. 2022. ALL RIGHTS
 */

#include <rdma/fi_cm.h>
#include <uct/base/uct_iface.h>
#include "ofi_iface.h"
#include "ofi_ep.h"


static ucs_config_field_t uct_ofi_iface_config_table[] = {
    {"", "ALLOC=huge,mmap,heap", NULL,
     ucs_offsetof(uct_ofi_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

     UCT_IFACE_MPOOL_CONFIG_FIELDS("OFI", -1, 0, "ofi",
                                   ucs_offsetof(uct_ofi_iface_config_t, mpool),
                                   "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
                                   "and could cause deadlock or performance degradation."),

    {NULL}
};

/* Note: This isn't thread safe. Does it need to be? */
/* TODO: Evaluate inlining this */
int uct_ofi_get_next_av(uct_ofi_av_t *av)
{
    int idx = UCS_BITMAP_FFS(av->free);
    if (idx > UCT_OFI_EPS_PER_AV) {
        /* TODO: Is opening more AVs a valid way of handling this? */
        ucs_error("Exceeded UCT_OFI_EPS_PER_AV!");
        return idx;
    }
    UCS_BITMAP_UNSET(av->free, idx);
    return idx;
}


void uct_ofi_free_av(uct_ofi_av_t *av, int idx)
{
    UCS_BITMAP_SET(av->free, idx);
}


ucs_status_t uct_ofi_iface_get_address(uct_iface_h tl_iface,
                                 uct_iface_addr_t *tl_addr)
{
    uct_ofi_iface_t *iface = ucs_derived_of(tl_iface, uct_ofi_iface_t);
    uct_ofi_name_t *addr = (uct_ofi_name_t *)tl_addr;
    int status;

    ucs_debug("OFI get iface address");
    
    status = fi_getname((fid_t)iface->local, &addr->name, &addr->size);
    if( !status ) {
        return UCS_OK;
    } else {
        return UCS_ERR_NO_DEVICE;
    }
}


int uct_ofi_iface_is_reachable(uct_iface_h tl_iface, const uct_device_addr_t *dev_addr, const uct_iface_addr_t *iface_addr)
{
    struct fi_info hints = {0};
    struct fi_info *info;
    uct_ofi_name_t *addr = (uct_ofi_name_t *)iface_addr;
    int ret;
    
    hints.caps = FI_RMA | FI_ATOMIC | FI_TAGGED;

    ucs_debug("OFI iface reachable");
    ret = fi_getinfo(fi_version(), addr->name, NULL, 0, &hints, &info);

    if (ret == -FI_ENODATA) {
        ucs_trace("OFI could not reach address");
        return 0;
    } else {
        ucs_trace("OFI can reach this address");
        fi_freeinfo(info);
        return 1;
    }
}


static ucs_status_t uct_ofi_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ofi_iface_t *iface = ucs_derived_of(tl_iface, uct_ofi_iface_t);
    uct_base_iface_query(&iface->super, iface_attr);
    ucs_trace("Query OFI");
    
    iface_attr->cap.am.max_short       = 0;
    iface_attr->cap.am.max_bcopy       = 0;
    iface_attr->cap.am.opt_zcopy_align = 1;
    iface_attr->cap.am.align_mtu       = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->device_addr_len        = 0;
    iface_attr->iface_addr_len         = sizeof(uct_ofi_name_t);
    iface_attr->ep_addr_len            = sizeof(uct_ofi_name_t);
    iface_attr->max_conn_priv          = 0;
    iface_attr->cap.flags              = 0;
    iface_attr->overhead               = 1e-6;  /* 1 usec */
    iface_attr->latency                = ucs_linear_func_make(40e-6, 0); /* 40 usec */
    iface_attr->bandwidth.dedicated    = 1.0 * UCS_MBYTE; /* bytes */
    iface_attr->bandwidth.shared       = 0;
    iface_attr->priority               = 0;
    
    return UCS_OK;
}


ucs_status_t uct_ofi_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp)
{
    return  UCS_ERR_UNSUPPORTED;
}

static ucs_status_t handle_cq_error(struct fid_cq *cq, int ret)
{
    /* TODO: More sophisticated cq handling */
    /* This function should return UCS_OK to tell the progress
       thread to continue and UCS_ERR_* to stop progress. */
    UCT_OFI_CHECK_ERROR(ret, "Err reading CQ", UCS_OK);
    return UCS_OK;
}

static int progress_cq(struct fid_cq *cq)
{
    struct fi_cq_entry entry;
    int ret, count=-1;
    
    do {
        count++;
        ret = fi_cq_read(cq, &entry, 1);
        if (ret < 0 && ret != -FI_EAGAIN) {
            handle_cq_error(cq, ret);
        }
    } while (ret != -FI_EAGAIN);

    return count;
}

static unsigned uct_ofi_progress(void *arg)
{
    int count = 0;
    uct_ofi_iface_t *iface = (uct_ofi_iface_t *)arg;

    count = progress_cq(iface->tx_cq);
    count += progress_cq(iface->rx_cq);

    return count;
}

static void clean_av(uct_ofi_iface_t *iface)
{
    fi_close(&iface->av->av->fid);
}

static void clean_cq(uct_ofi_iface_t *iface)
{
    fi_close(&iface->tx_cq->fid);
    fi_close(&iface->rx_cq->fid);
}

static void clean_ep(uct_ofi_iface_t *iface)
{
    fi_close(&iface->local->fid);
}

static void clean_info(uct_ofi_iface_t *iface)
{
    fi_freeinfo(iface->info);
}

void uct_ofi_cleanup_base_iface(uct_ofi_iface_t *iface)
{
    clean_ep(iface);
    clean_cq(iface);
    clean_av(iface);
    clean_info(iface);
}

static UCS_CLASS_CLEANUP_FUNC(uct_ofi_iface_t)
{
    uct_ofi_cleanup_base_iface(self);
}

extern ucs_class_t UCS_CLASS_DECL_NAME(uct_ofi_iface_t);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ofi_iface_t, uct_iface_t);

static uct_iface_ops_t uct_ofi_iface_ops = {
    .ep_pending_add           = uct_ofi_ep_pending_add,
    .ep_pending_purge         = uct_ofi_ep_pending_purge,
    .ep_flush                 = uct_ofi_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_ofi_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_ofi_ep_t),
    .ep_get_address           = uct_ofi_ep_get_address,
    .iface_flush              = uct_ofi_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (void*)uct_ofi_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ofi_iface_t),
    .iface_query              = uct_ofi_iface_query,
    .iface_get_device_address = uct_ofi_iface_get_dev_address,
    .iface_get_address        = uct_ofi_iface_get_address,
    .iface_is_reachable       = uct_ofi_iface_is_reachable
};

static uct_iface_internal_ops_t uct_ofi_iface_internal_ops = {
    .iface_estimate_perf = uct_base_iface_estimate_perf,
    .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query            = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate       = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported    
};


static ucs_status_t uct_ofi_setup_av(uct_ofi_md_t *md, uct_ofi_iface_t *iface)
{
    struct fi_av_attr av_attr = {0};
    int status;

    iface->av = ucs_malloc(sizeof(uct_ofi_av_t), "Address vector metadata");
    iface->av->table = ucs_malloc(sizeof(fi_addr_t) * UCT_OFI_EPS_PER_AV, "Address vector");
    av_attr.type = FI_AV_MAP;
    status = fi_av_open(md->dom_ctx,
                        &av_attr,
                        &iface->av->av,
                        NULL);
    if (status) {
        return UCS_ERR_NO_RESOURCE;
    }
    
    status = fi_ep_bind(iface->local, &iface->av->av->fid, 0);
    if(!status) {
        UCS_BITMAP_SET_ALL(iface->av->free);
        return UCS_OK;
    } else {
        return UCS_ERR_NO_RESOURCE;
    }    
}

static ucs_status_t uct_ofi_setup_target(uct_ofi_md_t *md, uct_ofi_iface_t *iface)
{
    int status;
    ucs_status_t ret = UCS_OK;

    status = fi_endpoint(md->dom_ctx, iface->info, &iface->local, NULL);

    if ( status ) {
        ret = UCS_ERR_NO_RESOURCE;
    }
    
    return ret;
}

static ucs_status_t uct_ofi_setup_cqs(uct_ofi_md_t *md, uct_ofi_iface_t *iface)
{
    struct fi_cq_attr attr = {0};
    int status;

    attr.format = FI_CQ_FORMAT_DATA;
    attr.size = 0;
    attr.wait_obj  = FI_WAIT_NONE;

    ucs_trace("Opening CQs");
    status = fi_cq_open(md->dom_ctx, &attr, &iface->tx_cq, NULL);
    if (status) {
        ucs_error("Could not open CQ: %s", fi_strerror(status));
        return UCS_ERR_NO_RESOURCE;
    }

    status = fi_cq_open(md->dom_ctx, &attr, &iface->rx_cq, NULL);
    if (status) {
        ucs_error("Could not open CQ: %s", fi_strerror(status));
        return UCS_ERR_NO_RESOURCE;
    }

    ucs_trace("Binding CQs");
    status = fi_ep_bind(iface->local, &iface->tx_cq->fid, FI_TRANSMIT);
    if (status) {
        ucs_error("Could not open CQ: %s", fi_strerror(status));
        return UCS_ERR_NO_RESOURCE;
    }

    status = fi_ep_bind(iface->local, &iface->rx_cq->fid, FI_RECV);
    if (status) {
        ucs_error("Could not open CQ: %s", fi_strerror(status));
        return UCS_ERR_NO_RESOURCE;
    }
    
    return UCS_OK;
}

static ucs_status_t uct_ofi_setup_fi_info(uct_ofi_md_t *md, uct_ofi_iface_t *iface, const uct_iface_params_t *params)
{
    struct fi_info hints = {0};
    int ret;
    
    hints.caps = FI_RMA | FI_ATOMIC | FI_TAGGED;
    ret = fi_getinfo(fi_version(), NULL, NULL, 0, &hints, &iface->info);
    if ( ret ) {
        /* TODO: Better return codes? */
        return UCS_ERR_NO_MEMORY;
    } else {
        return UCS_OK;
    }
}

UCS_CLASS_INIT_FUNC(uct_ofi_iface_t, uct_md_h tl_md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config
                    )
{
    ucs_status_t ret;
    uct_ofi_md_t *md = ucs_derived_of(tl_md, uct_ofi_md_t);
    
    ucs_debug("OFI init iface");
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_ofi_iface_ops, &uct_ofi_iface_internal_ops, tl_md,
                              worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG("OFI_MD"));

    ret = uct_ofi_setup_fi_info(md, self, params);
    if( ret != UCS_OK ) {
        goto out;
    }
    ucs_trace("OFI info allocated");
    ret = uct_ofi_setup_target(md, self);
    if( ret != UCS_OK ) {
        goto out_info;
    }
    ucs_trace("OFI target ep setup");
    ret = uct_ofi_setup_cqs(md, self);
    if( ret != UCS_OK ) {
        goto out_ep;
    }
    ucs_trace("OFI cqs setup");
    ret = uct_ofi_setup_av(md, self);
    if( ret != UCS_OK ) {
        goto out_cq;
    }
    ucs_debug("OFI iface creation successful");
    return UCS_OK;
 
 out_cq:
    clean_cq(self);
 out_ep:
    clean_ep(self);
 out_info:
    clean_info(self);
 out:
    return ret;
}

UCS_CLASS_DEFINE(uct_ofi_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ofi_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*,
                          const uct_iface_config_t *);
UCT_TL_DEFINE(&uct_ofi_component, ofi, uct_ofi_query_devices,
              uct_ofi_iface_t, "OFI_",
              uct_ofi_iface_config_table, uct_ofi_iface_config_t);
