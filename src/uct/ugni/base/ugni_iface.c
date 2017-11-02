/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_types.h"
#include "ugni_md.h"
#include "ugni_device.h"
#include "ugni_ep.h"
#include "ugni_iface.h"

ucs_status_t uct_ugni_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (0 == iface->outstanding) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface,
                                        uct_iface_addr_t *addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    uct_sockaddr_ugni_t *iface_addr = (uct_sockaddr_ugni_t*)addr;

    iface_addr->domain_id   = iface->cdm.domain_id;
    return UCS_OK;
}

int uct_ugni_iface_is_reachable(uct_iface_h tl_iface, const uct_device_addr_t *dev_addr, const uct_iface_addr_t *iface_addr)
{
    return 1;
}

static ucs_mpool_ops_t uct_ugni_flush_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

void uct_ugni_cleanup_base_iface(uct_ugni_iface_t *iface)
{
    ucs_arbiter_cleanup(&iface->arbiter);
    ucs_mpool_cleanup(&iface->flush_pool, 1);
    uct_ugni_destroy_cq(iface->local_cq, &iface->cdm);
    uct_ugni_destroy_cdm(&iface->cdm);
}

UCS_CLASS_INIT_FUNC(uct_ugni_iface_t, uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    uct_iface_ops_t *uct_ugni_iface_ops,
                    const uct_iface_config_t *tl_config
                    UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
    uct_ugni_device_t *dev;
    ucs_status_t status;
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    unsigned grow =  (config->mpool.bufs_grow == 0) ? 128 : config->mpool.bufs_grow;

    ucs_assert(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, uct_ugni_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_UGNI_MD_NAME));
    dev = uct_ugni_device_by_name(params->mode.device.dev_name);
    if (NULL == dev) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }
    status = uct_ugni_create_cdm(&self->cdm, dev, self->super.worker->thread_mode);
    if (UCS_OK != status) {
        ucs_error("Failed to UGNI NIC, Error status: %d", status);
        return status;
    }
    status = uct_ugni_create_cq(&self->local_cq, UCT_UGNI_LOCAL_CQ, &self->cdm);
    if (UCS_OK != status) {
        goto clean_cdm;
    }
    self->outstanding = 0;
    sglib_hashed_uct_ugni_ep_t_init(self->eps);
    ucs_arbiter_init(&self->arbiter);
    status = ucs_mpool_init(&self->flush_pool,
                            0,
                            sizeof(uct_ugni_flush_group_t),
                            0,                            /* alignment offset */
                            UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                            grow,                         /* grow */
                            config->mpool.max_bufs,       /* max buffers */
                            &uct_ugni_flush_mpool_ops,
                            "UGNI-DESC-ONLY");
    if (UCS_OK != status) {
        ucs_error("Could not init iface");
        goto clean_cq;
    }
    return status;
clean_cq:
    uct_ugni_destroy_cq(self->local_cq, &self->cdm);
clean_cdm:
    uct_ugni_destroy_cdm(&self->cdm);
    return status;
}

UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, uct_iface_ops_t *,
                          const uct_iface_config_t * UCS_STATS_ARG(ucs_stats_node_t *));

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_iface_t)
{
    uct_ugni_cleanup_base_iface(self);
}

UCS_CLASS_DEFINE(uct_ugni_iface_t, uct_base_iface_t);
