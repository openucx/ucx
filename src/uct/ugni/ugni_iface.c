/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include <pmi.h>
#include "ucs/type/class.h"
#include "uct/tl/context.h"

#include "ugni_iface.h"
#include "ugni_ep.h"

/* Forward declaration */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_iface_t)(uct_iface_t*);
static ucs_status_t uct_ugni_query_tl_resources(uct_pd_h pd,
                                                uct_tl_resource_desc_t **resource_p,
                                                unsigned *num_resources_p);
unsigned ugni_domain_global_counter = 0;

static void uct_ugni_progress(void *arg)
{
    gni_cq_entry_t  event_data = 0;
    gni_post_descriptor_t *event_post_desc_ptr;
    uct_ugni_base_desc_t *desc;
    uct_ugni_iface_t * iface = (uct_ugni_iface_t *)arg;
    gni_return_t ugni_rc;

    ugni_rc = GNI_CqGetEvent(iface->local_cq, &event_data);
    if (GNI_RC_NOT_DONE == ugni_rc) {
        return;
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
        uct_invoke_completion(desc->comp_cb);
    }
    --iface->outstanding;
    --desc->ep->outstanding;

    if (ucs_likely(desc->not_ready_to_free == 0)) {
        ucs_mpool_put(desc);
    }
    return;
}

static ucs_status_t uct_ugni_iface_flush(uct_iface_h tl_iface)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    if (0 == iface->outstanding) {
        return UCS_OK;
    }
    uct_ugni_progress(iface);
    return UCS_ERR_NO_RESOURCE;
}

static ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface,
                                               struct sockaddr *addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    uct_sockaddr_ugni_t *iface_addr = (uct_sockaddr_ugni_t*)addr;

    iface_addr->sgni_family = UCT_AF_UGNI;
    iface_addr->nic_addr    = iface->nic_addr;
    iface_addr->domain_id   = iface->domain_id;
    return UCS_OK;
}

static int uct_ugni_iface_is_reachable(uct_iface_h tl_iface, const struct sockaddr *addr)
{
    const uct_sockaddr_ugni_t *iface_addr = (const uct_sockaddr_ugni_t*)addr;

    if (iface_addr->sgni_family != UCT_AF_UGNI) {
        return 0;
    }

    return 1;
}

ucs_status_t uct_ugni_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    iface_attr->cap.put.max_short      = iface->config.fma_seg_size;
    iface_attr->cap.put.max_bcopy      = iface->config.fma_seg_size;
    iface_attr->cap.put.max_zcopy      = iface->config.rdma_max_size;
    iface_attr->cap.get.max_bcopy      = iface->config.fma_seg_size - 8; /* alignment offset 4 (addr)+ 4 (len)*/
    iface_attr->cap.get.max_zcopy      = iface->config.rdma_max_size;
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_ugni_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT |
                                         UCT_IFACE_FLAG_PUT_BCOPY |
                                         UCT_IFACE_FLAG_PUT_ZCOPY |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP64 |
                                         UCT_IFACE_FLAG_ATOMIC_FADD64  |
                                         UCT_IFACE_FLAG_ATOMIC_ADD64   |
                                         UCT_IFACE_FLAG_GET_BCOPY      |
                                         UCT_IFACE_FLAG_GET_ZCOPY      |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    return UCS_OK;
}

uct_iface_ops_t uct_ugni_iface_ops = {
    .iface_query         = uct_ugni_iface_query,
    .iface_flush         = uct_ugni_iface_flush,
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_iface_t),
    .iface_get_address   = uct_ugni_iface_get_address,
    .iface_is_reachable  = uct_ugni_iface_is_reachable,
    .ep_create_connected = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_ep_t),
    .ep_put_short        = uct_ugni_ep_put_short,
    .ep_put_bcopy        = uct_ugni_ep_put_bcopy,
    .ep_put_zcopy        = uct_ugni_ep_put_zcopy,
    .ep_am_short         = uct_ugni_ep_am_short,
    .ep_atomic_add64     = uct_ugni_ep_atomic_add64,
    .ep_atomic_fadd64    = uct_ugni_ep_atomic_fadd64,
    .ep_atomic_cswap64   = uct_ugni_ep_atomic_cswap64,
    .ep_get_bcopy        = uct_ugni_ep_get_bcopy,
    .ep_get_zcopy        = uct_ugni_ep_get_zcopy,
};

static void uct_ugni_base_desc_init(void *mp_context, void *obj, void *chunk, void *arg)
{
    uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *) obj;
    /* zero base descriptor */
    memset(base, 0 , sizeof(*base));
}

static void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh)
{
    uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *)obj;
    /* call base initialization */
    uct_ugni_base_desc_init(iface, obj, NULL, NULL);
    /* set local keys */
    base->desc.local_mem_hndl = *(gni_mem_handle_t *)memh;
}

static UCS_CLASS_INIT_FUNC(uct_ugni_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    uct_ugni_device_t *dev;
    ucs_status_t rc;

    pthread_mutex_lock(&uct_ugni_global_lock);

    dev = uct_ugni_device_by_name(dev_name);
    if (NULL == dev) {
        ucs_error("No device was found: %s", dev_name);
        rc = UCS_ERR_NO_DEVICE;
        goto exit;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_ugni_iface_ops, pd, worker,
                              &config->super UCS_STATS_ARG(NULL));

    self->dev      = dev;
    self->nic_addr = dev->address;

    /* Setting initial configuration */
    self->config.fma_seg_size  = UCT_UGNI_MAX_FMA;
    self->config.rdma_max_size = UCT_UGNI_MAX_RDMA;

    rc = ucs_mpool_create("UGNI-DESC-ONLY", sizeof(uct_ugni_base_desc_t),
                          0,                            /* alignment offset */
                          UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                          128 ,                         /* grow */
                          config->mpool.max_bufs,       /* max buffers */
                          &self->super.super,           /* iface */
                          ucs_mpool_hugetlb_malloc,     /* allocation hooks */
                          ucs_mpool_hugetlb_free,       /* free hook */
                          uct_ugni_base_desc_init,      /* init func */
                          NULL , &self->free_desc);
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto exit;
    }

    rc = ucs_mpool_create("UGNI-GET-DESC-ONLY", sizeof(uct_ugni_fetch_desc_t),
                          0,                            /* alignment offset */
                          UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                          128 ,                         /* grow */
                          config->mpool.max_bufs,       /* max buffers */
                          &self->super.super,           /* iface */
                          ucs_mpool_hugetlb_malloc,     /* allocation hooks */
                          ucs_mpool_hugetlb_free,       /* free hook */
                          uct_ugni_base_desc_init,      /* init func */
                          NULL , &self->free_desc_get);
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto clean_desc;
    }

    rc = ucs_mpool_create("UGNI-DESC-BUFFER",
                          sizeof(uct_ugni_base_desc_t) +
                          self->config.fma_seg_size,
                          sizeof(uct_ugni_base_desc_t), /* alignment offset */
                          UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                          128 ,                         /* grow */
                          config->mpool.max_bufs,       /* max buffers */
                          &self->super.super,           /* iface */
                          ucs_mpool_hugetlb_malloc,     /* allocation hooks */
                          ucs_mpool_hugetlb_free,       /* free hook */
                          uct_ugni_base_desc_init,      /* init func */
                          NULL , &self->free_desc_buffer);
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto clean_desc_get;
    }

    rc = uct_iface_mpool_create(&self->super.super,
                                sizeof(uct_ugni_fetch_desc_t) + 8,
                                sizeof(uct_ugni_fetch_desc_t),  /* alignment offset */
                                UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                                &config->mpool,               /* mpool config */
                                128 ,                         /* grow */
                                uct_ugni_base_desc_key_init,  /* memory/key init */
                                "UGNI-DESC-FAMO",             /* name */
                                &self->free_desc_famo);       /* mpool */
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto clean_buffer;
    }

    rc = uct_iface_mpool_create(&self->super.super,
                                sizeof(uct_ugni_fetch_desc_t) +
                                self->config.fma_seg_size,
                                sizeof(uct_ugni_fetch_desc_t), /* alignment offset */
                                UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                                &config->mpool,               /* mpool config */
                                128 ,                         /* grow */
                                uct_ugni_base_desc_key_init,  /* memory/key init */
                                "UGNI-DESC-GET",              /* name */
                                &self->free_desc_get_buffer); /* mpool */
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto clean_famo;
    }

    ucs_notifier_chain_add(&worker->progress_chain, uct_ugni_progress, self);

    self->activated = false;
    self->outstanding = 0;
    rc = ugni_activate_iface(self);
    if (UCS_OK == rc) {
        goto exit;
    }

    ucs_error("Failed to activate interface");

    ucs_mpool_destroy(self->free_desc_get_buffer);
clean_famo:
    ucs_mpool_destroy(self->free_desc_famo);
clean_buffer:
    ucs_mpool_destroy(self->free_desc_buffer);
clean_desc_get:
    ucs_mpool_destroy(self->free_desc_get);
clean_desc:
    ucs_mpool_destroy(self->free_desc);
exit:
    pthread_mutex_unlock(&uct_ugni_global_lock);
    return rc;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_iface_t)
{
    gni_return_t ugni_rc;

    ucs_notifier_chain_remove(&self->super.worker->progress_chain,
                              uct_ugni_progress, self);

    if (!self->activated) {
        /* We done with release */
        return;
    }

    ucs_mpool_destroy(self->free_desc_get_buffer);
    ucs_mpool_destroy(self->free_desc_get);
    ucs_mpool_destroy(self->free_desc_famo);
    ucs_mpool_destroy(self->free_desc_buffer);
    ucs_mpool_destroy(self->free_desc);

    /* TBD: Clean endpoints first (unbind and destroy) ?*/
    ugni_rc = GNI_CqDestroy(self->local_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CqDestroy failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
    }
    ugni_rc = GNI_CdmDestroy(self->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CdmDestroy error status: %s (%d)",
                 gni_err_str[ugni_rc], ugni_rc);
    }
    self->activated = false;
}

UCS_CLASS_DEFINE(uct_ugni_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_iface_t, uct_iface_t,
                          uct_pd_h, uct_worker_h,
                          const char*, size_t, const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_iface_t, uct_iface_t);

UCT_TL_COMPONENT_DEFINE(uct_ugni_tl_component,
                        uct_ugni_query_tl_resources,
                        uct_ugni_iface_t,
                        UCT_UGNI_TL_NAME,
                        "UGNI_",
                        uct_ugni_iface_config_table,
                        uct_ugni_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ugni_pd_component, &uct_ugni_tl_component);

static ucs_status_t uct_ugni_query_tl_resources(uct_pd_h pd,
                                                uct_tl_resource_desc_t **resource_p,
                                                unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resources;
    int num_devices = job_info.num_devices;
    uct_ugni_device_t *devs = job_info.devices;
    int i;
    ucs_status_t rc = UCS_OK;

    assert(!strncmp(pd->component->name,
                    UCT_UGNI_TL_NAME,
                    UCT_PD_NAME_MAX));

    pthread_mutex_lock(&uct_ugni_global_lock);

    resources = ucs_calloc(job_info.num_devices, sizeof(uct_tl_resource_desc_t),
                          "resource desc");
    if (NULL == resources) {
      ucs_error("Failed to allocate memory");
      num_devices = 0;
      resources = NULL;
      rc = UCS_ERR_NO_MEMORY;
      goto error;
    }

    for (i = 0; i < job_info.num_devices; i++) {
        uct_ugni_device_get_resource(&devs[i], &resources[i]);
    }

error:
    *num_resources_p = num_devices;
    *resource_p      = resources;
    pthread_mutex_unlock(&uct_ugni_global_lock);

    return rc;
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

static ucs_status_t uct_ugni_fetch_pmi()
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

ucs_status_t uct_ugni_init_nic(int device_index,
                               int *domain_id,
                               gni_cdm_handle_t *cdm_handle,
                               gni_nic_handle_t *nic_handle,
                               uint32_t *address)
{
    int modes;
    ucs_status_t rc;
    gni_return_t ugni_rc = GNI_RC_SUCCESS;

    rc = uct_ugni_fetch_pmi();
    if (UCS_OK != rc) {
        ucs_error("Failed to activate context, Error status: %d", rc);
        return rc;
    }

    *domain_id = job_info.pmi_rank_id + job_info.pmi_num_of_ranks * ugni_domain_global_counter;
    modes = GNI_CDM_MODE_FORK_FULLCOPY | GNI_CDM_MODE_CACHED_AMO_ENABLED |
        GNI_CDM_MODE_ERR_NO_KILL | GNI_CDM_MODE_FAST_DATAGRAM_POLL;
    ucs_debug("Creating new PD domain with id %d (%d + %d * %d)",
              *domain_id, job_info.pmi_rank_id,
              job_info.pmi_num_of_ranks, ugni_domain_global_counter);
    ugni_rc = GNI_CdmCreate(*domain_id, job_info.ptag, job_info.cookie,
                            modes, cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    /* For now we use the first device for allocation of the domain */
    ugni_rc = GNI_CdmAttach(*cdm_handle, job_info.devices[device_index].device_id,
                            address, nic_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmAttach failed (domain id %d, %d), Error status: %s %d",
                  *domain_id, ugni_domain_global_counter, gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    ++ugni_domain_global_counter;
    return UCS_OK;
}

#define UCT_UGNI_LOCAL_CQ (8192)

ucs_status_t ugni_activate_iface(uct_ugni_iface_t *iface)
{
    int rc;
    gni_return_t ugni_rc;

    if(iface->activated) {
        return UCS_OK;
    }

    rc = uct_ugni_init_nic(0, &iface->domain_id,
                           &iface->cdm_handle, &iface->nic_handle,
                           &iface->pe_address);
    if (UCS_OK != rc) {
        ucs_error("Failed to UGNI NIC, Error status: %d", rc);
        return rc;
    }

    ugni_rc = GNI_CqCreate(iface->nic_handle, UCT_UGNI_LOCAL_CQ, 0,
                           GNI_CQ_NOBLOCK,
                           NULL, NULL, &iface->local_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CqCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }
    iface->activated = true;

    /* iface is activated */
    return UCS_OK;
}
