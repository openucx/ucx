#include "ugni_iface.h"
#include <pmi.h>

static unsigned ugni_domain_global_counter = 0;

void uct_ugni_base_desc_init(void *mp_context, void *obj, void *chunk, void *arg)
{
  uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *) obj;
  /* zero base descriptor */
  memset(base, 0 , sizeof(*base));
}

void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh)
{
  uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *)obj;
  /* call base initialization */
  uct_ugni_base_desc_init(iface, obj, NULL, NULL);
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

ucs_status_t uct_ugni_iface_flush(uct_iface_h tl_iface)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    if (0 == iface->outstanding) {
        return UCS_OK;
    }
    uct_ugni_progress(iface);
    return UCS_ERR_NO_RESOURCE;
}

ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface,
                                               struct sockaddr *addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    uct_sockaddr_ugni_t *iface_addr = (uct_sockaddr_ugni_t*)addr;

    iface_addr->sgni_family = UCT_AF_UGNI;
    iface_addr->nic_addr    = iface->nic_addr;
    iface_addr->domain_id   = iface->domain_id;
    return UCS_OK;
}

int uct_ugni_iface_is_reachable(uct_iface_h tl_iface, const struct sockaddr *addr)
{
    const uct_sockaddr_ugni_t *iface_addr = (const uct_sockaddr_ugni_t*)addr;

    if (iface_addr->sgni_family != UCT_AF_UGNI) {
        return 0;
    }

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

UCS_CLASS_INIT_FUNC(uct_ugni_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, uct_iface_ops_t *uct_ugni_iface_ops,
                           const uct_iface_config_t *tl_config
                           UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
  uct_ugni_device_t *dev;

  dev = uct_ugni_device_by_name(dev_name);
  if (NULL == dev) {
    ucs_error("No device was found: %s", dev_name);
    return UCS_ERR_NO_DEVICE;
  }

  UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, uct_ugni_iface_ops, pd, worker,
                            tl_config UCS_STATS_ARG(NULL));

  self->dev      = dev;
  self->nic_addr = dev->address;

  self->activated = false;
  self->outstanding = 0;

  return UCS_OK;
}

UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_iface_t, uct_iface_t,
                          uct_pd_h, uct_worker_h,
                          const char*, uct_iface_ops_t *, const uct_iface_config_t * UCS_STATS_ARG(ucs_stats_node_t *));
static UCS_CLASS_CLEANUP_FUNC(uct_ugni_iface_t)
{
}

UCS_CLASS_DEFINE(uct_ugni_iface_t, uct_base_iface_t);
