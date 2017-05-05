/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ugni_device.h"
#include "ugni_md.h"
#include "ugni_iface.h"
#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>

#if ENABLE_MT
#define uct_ugni_check_lock_needed(_cdm) UCS_THREAD_MODE_MULTI == _cdm->thread_mode
#define uct_ugni_device_init_lock(_dev) ucs_spinlock_init(&_dev->lock)
#define uct_ugni_device_destroy_lock(_dev) ucs_spinlock_destroy(&_dev->lock)
#define uct_ugni_device_lock(_cdm) \
if (uct_ugni_check_lock_needed(_cdm)) {  \
    ucs_spin_lock(&cdm->dev->lock);          \
}
#define uct_ugni_device_unlock(_cdm) \
if (uct_ugni_check_lock_needed(_cdm)) {    \
    ucs_spin_unlock(&cdm->dev->lock);          \
}
#else
#define uct_ugni_device_init_lock(x) UCS_OK
#define uct_ugni_device_destroy_lock(x) UCS_OK
#define uct_ugni_device_lock(x)
#define uct_ugni_device_unlock(x)
#define uct_ugni_check_lock_needed(x) 0
#endif

uint16_t ugni_domain_counter = 0;

void uct_ugni_device_get_resource(const char *tl_name, uct_ugni_device_t *dev,
                                  uct_tl_resource_desc_t *resource)
{
    ucs_snprintf_zero(resource->tl_name,  sizeof(resource->tl_name), "%s", tl_name);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s", dev->fname);
    resource->dev_type = UCT_DEVICE_TYPE_NET;
}

static ucs_status_t get_nic_address(uct_ugni_device_t *dev_p)
{
    int             alps_addr = -1;
    int             alps_dev_id = -1;
    int             i;
    char           *token, *pmi_env;

    pmi_env = getenv("PMI_GNI_DEV_ID");
    if (NULL == pmi_env) {
        gni_return_t ugni_rc;
        ugni_rc = GNI_CdmGetNicAddress(dev_p->device_id, &dev_p->address,
                                       &dev_p->cpu_id);
        if (GNI_RC_SUCCESS != ugni_rc) {
            ucs_error("GNI_CdmGetNicAddress failed, device %d, Error status: %s %d",
                      dev_p->device_id, gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_NO_DEVICE;
        }
        CPU_SET(dev_p->cpu_id, &(dev_p->cpu_mask));
        ucs_debug("(GNI) NIC address: %d", dev_p->address);
    } else {
        while ((token = strtok(pmi_env, ":")) != NULL) {
            alps_dev_id = atoi(token);
            if (alps_dev_id == dev_p->device_id) {
                break;
            }
            pmi_env = NULL;
        }
        ucs_assert(alps_dev_id != -1);

        pmi_env = getenv("PMI_GNI_LOC_ADDR");
        ucs_assert(NULL != pmi_env);
        i = 0;
        while ((token = strtok(pmi_env, ":")) != NULL) {
            if (i == alps_dev_id) {
                alps_addr = atoi(token);
                break;
            }
            pmi_env = NULL;
            ++i;
        }
        ucs_assert(alps_addr != -1);
        dev_p->address = alps_addr;
        ucs_debug("(PMI) NIC address: %d", dev_p->address);
    }
    return UCS_OK;
}

ucs_status_t uct_ugni_device_create(int dev_id, int index, uct_ugni_device_t *dev_p)
{
    ucs_status_t status;
    gni_return_t ugni_rc;

    dev_p->device_id = (uint32_t)dev_id;
    dev_p->device_index = index;

    status = get_nic_address(dev_p);
    if (UCS_OK != status) {
        ucs_error("Failed to get NIC address");
        return status;
    }

    ugni_rc = GNI_GetDeviceType(&dev_p->type);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_GetDeviceType failed, device %d, Error status: %s %d",
                  dev_id, gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    switch (dev_p->type) {
    case GNI_DEVICE_GEMINI:
        ucs_snprintf_zero(dev_p->type_name, sizeof(dev_p->type_name), "%s",
                          "GEMINI");
        break;
    case GNI_DEVICE_ARIES:
        ucs_snprintf_zero(dev_p->type_name, sizeof(dev_p->type_name), "%s",
                          "ARIES");
        break;
    default:
        ucs_snprintf_zero(dev_p->type_name, sizeof(dev_p->type_name), "%s",
                          "UNKNOWN");
    }

    ucs_snprintf_zero(dev_p->fname, sizeof(dev_p->fname), "%s:%d",
                      dev_p->type_name, dev_p->device_index);

    status = uct_ugni_device_init_lock(dev_p);
    if (UCS_OK != status) {
        ucs_error("Couldn't initalize device lock.");
        return status;
    }
    dev_p->attached = false;
    return UCS_OK;
}

void uct_ugni_device_destroy(uct_ugni_device_t *dev)
{
    ucs_status_t status;

    status = uct_ugni_device_destroy_lock(dev);
    if (UCS_OK != status) {
        ucs_error("Couldn't destroy device lock.");
    }
}

ucs_status_t uct_ugni_iface_get_dev_address(uct_iface_t *tl_iface, uct_device_addr_t *addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    uct_devaddr_ugni_t *ugni_dev_addr = (uct_devaddr_ugni_t *)addr;
    uct_ugni_device_t *dev = uct_ugni_iface_device(iface);

    ugni_dev_addr->nic_addr = dev->address;

    return UCS_OK;
}

ucs_status_t uct_ugni_create_cdm(uct_ugni_cdm_t *cdm, uct_ugni_device_t *device, ucs_thread_mode_t thread_mode)
{
    uct_ugni_job_info_t *job_info;
    int modes;
    gni_return_t ugni_rc;
    ucs_status_t status = UCS_OK;

    job_info = uct_ugni_get_job_info();
    if (NULL == job_info) {
        return UCS_ERR_IO_ERROR;
    }

    cdm->thread_mode = thread_mode;
    cdm->dev = device;
    uct_ugni_device_lock(cdm);
    cdm->domain_id = job_info->pmi_rank_id + job_info->pmi_num_of_ranks * ugni_domain_counter++;
    ucs_debug("Creating new command domain with id %d (%d + %d * %d)",
              cdm->domain_id, job_info->pmi_rank_id,
              job_info->pmi_num_of_ranks, ugni_domain_counter);
    modes = GNI_CDM_MODE_FORK_FULLCOPY | GNI_CDM_MODE_CACHED_AMO_ENABLED |
        GNI_CDM_MODE_ERR_NO_KILL | GNI_CDM_MODE_FAST_DATAGRAM_POLL;
    ugni_rc = GNI_CdmCreate(cdm->domain_id, job_info->ptag, job_info->cookie,
                            modes, &cdm->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_NO_DEVICE;
        goto out_unlock;
    }

    ugni_rc = GNI_CdmAttach(cdm->cdm_handle, device->device_id,
                            &cdm->address, &cdm->nic_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmAttach failed (domain id %d, %d), Error status: %s %d",
                  cdm->domain_id, ugni_domain_counter, gni_err_str[ugni_rc], ugni_rc);
        GNI_CdmDestroy(cdm->cdm_handle);
        status = UCS_ERR_NO_DEVICE;
    }

out_unlock:
    uct_ugni_device_unlock(cdm);
    if (UCS_OK == status) {
        ucs_debug("Made ugni cdm. nic_addr = %i domain_id = %i", device->address, cdm->domain_id);
    }
    return status;
}

ucs_status_t uct_ugni_destroy_cdm(uct_ugni_cdm_t *cdm)
{
    gni_return_t ugni_rc;

    ucs_debug("MD GNI_CdmDestroy");
    ugni_rc = GNI_CdmDestroy(cdm->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmDestroy error status: %s (%d)",
                 gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}
