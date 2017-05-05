/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_types.h"
#include "ugni_md.h"
#include "ugni_device.h"
#include "ugni_ep.h"
#include "ugni_iface.h"
#include <pmi.h>

void uct_ugni_base_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
  uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *) obj;
  /* zero base descriptor */
  memset(base, 0 , sizeof(*base));
}

void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh)
{
  uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *)obj;
  /* call base initialization */
  uct_ugni_base_desc_init(NULL, obj, NULL);
  /* set local keys */
  base->desc.local_mem_hndl = *(gni_mem_handle_t *)memh;
}

void uct_ugni_progress(void *arg)
{
    gni_cq_entry_t  event_data = 0;
    gni_post_descriptor_t *event_post_desc_ptr;
    uct_ugni_base_desc_t *desc;
    uct_ugni_iface_t * iface = (uct_ugni_iface_t *)arg;
    gni_return_t ugni_rc;

    ugni_rc = GNI_CqGetEvent(iface->local_cq, &event_data);
    if (GNI_RC_NOT_DONE == ugni_rc) {
        goto out;
    }

    if ((GNI_RC_SUCCESS != ugni_rc && !event_data) || GNI_CQ_OVERRUN(event_data)) {
        ucs_error("GNI_CqGetEvent falied. Error status %s %d ",
                  gni_err_str[ugni_rc], ugni_rc);
        return;
    }

    ugni_rc = GNI_GetCompleted(iface->local_cq, event_data, &event_post_desc_ptr);
    if (GNI_RC_SUCCESS != ugni_rc && GNI_RC_TRANSACTION_ERROR != ugni_rc) {
        ucs_error("GNI_GetCompleted falied. Error status %s %d %d",
                  gni_err_str[ugni_rc], ugni_rc, GNI_RC_TRANSACTION_ERROR);
        return;
    }

    desc = (uct_ugni_base_desc_t *)event_post_desc_ptr;
    ucs_trace_async("Completion received on %p", desc);

    if (NULL != desc->comp_cb) {
        uct_invoke_completion(desc->comp_cb, UCS_OK);
    }

    if (ucs_likely(0 == desc->not_ready_to_free)) {
        ucs_mpool_put(desc);
    }

    iface->outstanding--;
    uct_ugni_check_flush(desc->flush_group);

out:
    /* have a go a processing the pending queue */
    ucs_arbiter_dispatch(&iface->arbiter, 1, uct_ugni_ep_process_pending, NULL);
    return;
}

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

ucs_status_t uct_ugni_query_tl_resources(uct_md_h md, const char *tl_name,
                                         uct_tl_resource_desc_t **resource_p,
                                         unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resources;
    int num_devices = job_info.num_devices;
    uct_ugni_device_t *devs = job_info.devices;
    int i;
    ucs_status_t status = UCS_OK;

    pthread_mutex_lock(&uct_ugni_global_lock);

    resources = ucs_calloc(job_info.num_devices, sizeof(uct_tl_resource_desc_t),
                          "resource desc");
    if (NULL == resources) {
      ucs_error("Failed to allocate memory");
      num_devices = 0;
      resources = NULL;
      status = UCS_ERR_NO_MEMORY;
      goto error;
    }

    for (i = 0; i < job_info.num_devices; i++) {
        uct_ugni_device_get_resource(tl_name, &devs[i], &resources[i]);
    }

error:
    *num_resources_p = num_devices;
    *resource_p      = resources;
    pthread_mutex_unlock(&uct_ugni_global_lock);

    return status;
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

static ucs_status_t get_cookie(uint32_t *cookie)
{
    char           *cookie_str;
    char           *cookie_token;

    cookie_str = getenv("PMI_GNI_COOKIE");
    if (NULL == cookie_str) {
        ucs_error("getenv PMI_GNI_COOKIE failed");
        return UCS_ERR_IO_ERROR;
    }

    cookie_token = strtok(cookie_str, ":");
    if (NULL == cookie_token) {
        ucs_error("Failed to read PMI_GNI_COOKIE token");
        return UCS_ERR_IO_ERROR;
    }

    *cookie = (uint32_t) atoi(cookie_token);
    return UCS_OK;
}

static ucs_status_t get_ptag(uint8_t *ptag)
{
    char           *ptag_str;
    char           *ptag_token;

    ptag_str = getenv("PMI_GNI_PTAG");
    if (NULL == ptag_str) {
        ucs_error("getenv PMI_GNI_PTAG failed");
        return UCS_ERR_IO_ERROR;
    }

    ptag_token = strtok(ptag_str, ":");
    if (NULL == ptag_token) {
        ucs_error("Failed to read PMI_GNI_PTAG token");
        return UCS_ERR_IO_ERROR;
    }

    *ptag = (uint8_t) atoi(ptag_token);
    return UCS_OK;
}

ucs_status_t uct_ugni_fetch_pmi()
{
    int spawned = 0,
        rc;

    if(job_info.initialized) {
        return UCS_OK;
    }

    /* Fetch information from Cray's PMI */
    rc = PMI_Init(&spawned);
    if (PMI_SUCCESS != rc) {
        ucs_error("PMI_Init failed, Error status: %d", rc);
        return UCS_ERR_IO_ERROR;
    }
    ucs_debug("PMI spawned %d", spawned);

    rc = PMI_Get_size(&job_info.pmi_num_of_ranks);
    if (PMI_SUCCESS != rc) {
        ucs_error("PMI_Get_size failed, Error status: %d", rc);
        return UCS_ERR_IO_ERROR;
    }
    ucs_debug("PMI size %d", job_info.pmi_num_of_ranks);

    rc = PMI_Get_rank(&job_info.pmi_rank_id);
    if (PMI_SUCCESS != rc) {
        ucs_error("PMI_Get_rank failed, Error status: %d", rc);
        return UCS_ERR_IO_ERROR;
    }
    ucs_debug("PMI rank %d", job_info.pmi_rank_id);

    rc = get_ptag(&job_info.ptag);
    if (UCS_OK != rc) {
        ucs_error("get_ptag failed, Error status: %d", rc);
        return rc;
    }
    ucs_debug("PMI ptag %d", job_info.ptag);

    rc = get_cookie(&job_info.cookie);
    if (UCS_OK != rc) {
        ucs_error("get_cookie failed, Error status: %d", rc);
        return rc;
    }
    ucs_debug("PMI cookie %d", job_info.cookie);

    /* Context and domain is activated */
    job_info.initialized = true;
    ucs_debug("UGNI job info was activated");
    return UCS_OK;
}

static ucs_mpool_ops_t uct_ugni_flush_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

UCS_CLASS_INIT_FUNC(uct_ugni_iface_t, uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    uct_iface_ops_t *uct_ugni_iface_ops,
                    const uct_iface_config_t *tl_config
                    UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
    uct_ugni_device_t *dev;
    gni_return_t ugni_rc;
    ucs_status_t status;
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    unsigned grow =  (config->mpool.bufs_grow == 0) ? 128 : config->mpool.bufs_grow;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, uct_ugni_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_UGNI_MD_NAME));
    dev = uct_ugni_device_by_name(params->dev_name);
    if (NULL == dev) {
        ucs_error("No device was found: %s", params->dev_name);
        return UCS_ERR_NO_DEVICE;
    }
    status = uct_ugni_create_cdm(&self->cdm, dev, worker->thread_mode);
    if (UCS_OK != status) {
        ucs_error("Failed to UGNI NIC, Error status: %d", status);
        return status;
    }
    ugni_rc = GNI_CqCreate(uct_ugni_iface_nic_handle(self), UCT_UGNI_LOCAL_CQ, 0,
                           GNI_CQ_NOBLOCK,
                           NULL, NULL, &self->local_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CqCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
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
    }
    return status;
}

UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, uct_iface_ops_t *,
                          const uct_iface_config_t * UCS_STATS_ARG(ucs_stats_node_t *));

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_iface_t)
{
    GNI_CqDestroy(self->local_cq);
    uct_ugni_destroy_cdm(&self->cdm);
    ucs_arbiter_cleanup(&self->arbiter);
}

UCS_CLASS_DEFINE(uct_ugni_iface_t, uct_base_iface_t);
