/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * Copyright (c) Los Alamos National Security, LLC. 2018. ALL RIGHTS RESERVED.
 * Copyright (c) Triad National Security, LLC. 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_device.h"
#include "ugni_md.h"
#include "ugni_iface.h"
#include <uct/base/uct_md.h>
#include <ucs/arch/atomic.h>
#include <ucs/sys/string.h>
#include <pmi.h>

/**
 * @breif Static information about UGNI job
 *
 * This is static information about Cray's job.
 * The information is static and does not change since job launch.
 * Therefore, the information is only fetched once.
 */
typedef struct uct_ugni_job_info {
    uint8_t             ptag;                           /**< Protection tag */
    uint32_t            cookie;                         /**< Unique identifier generated by the PMI system */
    int                 num_devices;                    /**< Number of devices */
    uct_ugni_device_t   devices[UCT_UGNI_MAX_DEVICES];  /**< Array of devices */
    int                 initialized;                    /**< Info status */
} uct_ugni_job_info_t;

static uct_ugni_job_info_t job_info = {
    .num_devices        = -1,
};

uint32_t ugni_domain_counter = 0;

void uct_ugni_device_get_resource(uct_ugni_device_t *dev,
                                  uct_tl_device_resource_t *tl_device)
{
    ucs_snprintf_zero(tl_device->name, sizeof(tl_device->name), "%s", dev->fname);
    tl_device->type = UCT_DEVICE_TYPE_NET;
}

ucs_status_t uct_ugni_query_devices(uct_md_h md,
                                    uct_tl_device_resource_t **tl_devices_p,
                                    unsigned *num_tl_devices_p)
{
    uct_tl_device_resource_t *resources;
    int num_devices = job_info.num_devices;
    uct_ugni_device_t *devs = job_info.devices;
    int i;
    ucs_status_t status = UCS_OK;

    resources = ucs_calloc(job_info.num_devices, sizeof(*resources),
                           "resource desc");
    if (NULL == resources) {
      ucs_error("Failed to allocate memory");
      num_devices = 0;
      resources = NULL;
      status = UCS_ERR_NO_MEMORY;
      goto error;
    }

    for (i = 0; i < job_info.num_devices; i++) {
        uct_ugni_device_get_resource(&devs[i], &resources[i]);
    }

error:
    *num_tl_devices_p = num_devices;
    *tl_devices_p     = resources;

    return status;
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

    if (NULL == getenv ("PMI_GNI_COOKIE")) {
        /* Fetch information from Cray's PMI if needed */
        rc = PMI_Init(&spawned);
        if (PMI_SUCCESS != rc) {
            ucs_error("PMI_Init failed, Error status: %d", rc);
            return UCS_ERR_IO_ERROR;
        }
        ucs_debug("PMI spawned %d", spawned);
    }

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
    job_info.initialized = 1;
    ucs_debug("UGNI job info was activated");
    return UCS_OK;
}

static uct_ugni_job_info_t *uct_ugni_get_job_info()
{
    ucs_status_t status;

    status = uct_ugni_fetch_pmi();
    if (UCS_OK != status) {
        ucs_error("Could not fetch PMI info.");
        return NULL;
    }
    return &job_info;
}

ucs_status_t init_device_list()
{
    ucs_status_t status = UCS_OK;
    int i, num_active_devices;
    int *dev_ids = NULL;
    gni_return_t ugni_rc = GNI_RC_SUCCESS;
    uct_ugni_job_info_t *inf = NULL;

    /* check if devices were already initilized */

    inf = uct_ugni_get_job_info();
    if (NULL == inf) {
        ucs_error("Unable to get Cray PMI info");
        status = UCS_ERR_IO_ERROR;
        goto err_zero;
    }

    if (-1 != inf->num_devices) {
        ucs_debug("The device list is already initialized");
        status = UCS_OK;
        goto err_zero;
    }

    ugni_rc = GNI_GetNumLocalDevices(&inf->num_devices);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_GetNumLocalDevices failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_NO_DEVICE;
        goto err_zero;
    }

    if (0 == inf->num_devices) {
        ucs_debug("UGNI No device found");
        status = UCS_OK;
        goto err_zero;
    }

    if (inf->num_devices >= UCT_UGNI_MAX_DEVICES) {
        ucs_error("UGNI, number of discovered devices (%d) " \
                  "is above the maximum supported devices (%d)",
                  inf->num_devices, UCT_UGNI_MAX_DEVICES);
        status = UCS_ERR_UNSUPPORTED;
        goto err_zero;
    }

    dev_ids = ucs_calloc(inf->num_devices, sizeof(int), "ugni device ids");
    if (NULL == dev_ids) {
        ucs_error("Failed to allocate memory");
        status = UCS_ERR_NO_MEMORY;
        goto err_zero;
    }

    ugni_rc = GNI_GetLocalDeviceIds(inf->num_devices, dev_ids);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_GetLocalDeviceIds failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_NO_DEVICE;
        goto err_dev_id;
    }

    num_active_devices = 0;
    for (i = 0; i < inf->num_devices; i++) {
        status = uct_ugni_device_create(dev_ids[i], num_active_devices, &inf->devices[i]);
        if (status != UCS_OK) {
            ucs_warn("Failed to initialize ugni device %d (%s), ignoring it",
                     i, ucs_status_string(status));
        } else {
            ++num_active_devices;
        }
    }

    if (num_active_devices != inf->num_devices) {
        ucs_warn("Error in detection devices");
        status = UCS_ERR_NO_DEVICE;
        goto err_dev_id;
    }

    ucs_debug("Initialized UGNI component with %d devices", inf->num_devices);

err_dev_id:
    ucs_free(dev_ids);
err_zero:
    return status;
}

uct_ugni_device_t *uct_ugni_device_by_name(const char *dev_name)
{
    uct_ugni_device_t *dev;
    unsigned dev_index;

    if ((NULL == dev_name)) {
        ucs_error("Bad parameter. Device name is set to NULL");
        return NULL;
    }

    for (dev_index = 0; dev_index < job_info.num_devices; ++dev_index) {
        dev = &job_info.devices[dev_index];
        if ((strlen(dev_name) == strlen(dev->fname)) &&
            (0 == strncmp(dev_name, dev->fname, strlen(dev->fname)))) {
            ucs_debug("Device found: %s", dev_name);
            return dev;
        }
    }

    /* Device not found */
    ucs_error("Cannot find: %s", dev_name);
    return NULL;
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

ucs_status_t uct_ugni_device_create(int dev_id, int idx, uct_ugni_device_t *dev_p)
{
    ucs_status_t status;
    gni_return_t ugni_rc;

    dev_p->device_id = (uint32_t)dev_id;

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
                      dev_p->type_name, idx);

    return UCS_OK;
}

void uct_ugni_device_destroy(uct_ugni_device_t *dev)
{
}

ucs_status_t uct_ugni_iface_get_dev_address(uct_iface_t *tl_iface, uct_device_addr_t *addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    uct_devaddr_ugni_t *ugni_dev_addr = (uct_devaddr_ugni_t *)addr;
    uct_ugni_device_t *dev = uct_ugni_iface_device(iface);

    ugni_dev_addr->nic_addr = dev->address;

    return UCS_OK;
}

static int uct_ugni_next_power_of_two_inclusive (int value)
{
    int i, j, bit;

    for (i = 3, bit = 31 ; i >= 0 ; --i) {
        if (!(value & (0xff << (i << 3)))) {
            /* short circuit. no set bits present in this byte */
            bit -= 8;
            continue;
        }

        for (j = 7 ; j >= 0 ; --j, --bit) {
            int tmp = (1 << bit);
            if (value & tmp) {
                return (value == tmp) ? bit : bit + 1;
            }
        }
    }

    return 0;
}

ucs_status_t uct_ugni_create_cdm(uct_ugni_cdm_t *cdm, uct_ugni_device_t *device, ucs_thread_mode_t thread_mode)
{
    uct_ugni_job_info_t *j_info;
    int modes;
    gni_return_t ugni_rc;
    ucs_status_t status = UCS_OK;
    int pid_max = 32768, free_bits;
    FILE *fh;

    j_info = uct_ugni_get_job_info();
    if (NULL == j_info) {
        return UCS_ERR_IO_ERROR;
    }

    fh = fopen ("/proc/sys/kernel/pid_max", "r");
    if (NULL != fh) {
        fscanf (fh, "%d", &pid_max);
        fclose (fh);
    }

    /* determine how many free bits we have in the PID space (10 (64-bit) or more (32-bit)) */
    free_bits = 31 - (uct_ugni_next_power_of_two_inclusive (pid_max) - 1);

    cdm->thread_mode = thread_mode;
    cdm->dev = device;
    /* don't colide with the btl/ugni CDM space if used in the same process. this is done by setting the
     * highest bit in the CDM identifier */
    cdm->domain_id = 0x80000000ul | ((getpid () << free_bits) + ucs_atomic_fadd32(&ugni_domain_counter, 1));
    ucs_debug("Creating new command domain with id 0x%08x (0x80000000ul | ((%d << %d) + %d))",
              cdm->domain_id, getpid (), free_bits, ugni_domain_counter);
    modes = GNI_CDM_MODE_FORK_FULLCOPY | GNI_CDM_MODE_CACHED_AMO_ENABLED |
        GNI_CDM_MODE_ERR_NO_KILL | GNI_CDM_MODE_FAST_DATAGRAM_POLL | GNI_CDM_MODE_FMA_SHARED;
    ugni_rc = GNI_CdmCreate(cdm->domain_id, j_info->ptag, j_info->cookie,
                            modes, &cdm->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    ugni_rc = GNI_CdmAttach(cdm->cdm_handle, device->device_id,
                            &cdm->address, &cdm->nic_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmAttach failed, Error status: %s\n"
                  "Created domain 0x%08x",
                  gni_err_str[ugni_rc], cdm->domain_id);
        uct_ugni_destroy_cdm(cdm);
        return UCS_ERR_NO_DEVICE;
    }

    status = uct_ugni_cdm_init_lock(cdm);
    if (UCS_OK != status) {
        ucs_error("Couldn't initalize CDM lock.");
    }

    if (UCS_OK == status) {
        ucs_debug("Made ugni cdm. nic_addr = %i domain_id = 0x%08x", device->address, cdm->domain_id);
    }
    return status;
}

ucs_status_t uct_ugni_create_md_cdm(uct_ugni_cdm_t *cdm)
{
    return uct_ugni_create_cdm(cdm, &job_info.devices[0], UCS_THREAD_MODE_MULTI);
}

ucs_status_t uct_ugni_destroy_cdm(uct_ugni_cdm_t *cdm)
{
    gni_return_t ugni_rc;

    uct_ugni_cdm_destroy_lock(cdm);

    ucs_trace_func("cdm=%p", cdm);
    ugni_rc = GNI_CdmDestroy(cdm->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmDestroy error status: %s (%d)",
                 gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

ucs_status_t uct_ugni_create_cq(gni_cq_handle_t *cq, unsigned cq_size, uct_ugni_cdm_t *cdm)
{
    gni_return_t ugni_rc;

    ugni_rc = GNI_CqCreate(cdm->nic_handle, UCT_UGNI_LOCAL_CQ, 0,
                           GNI_CQ_NOBLOCK,
                           NULL, NULL, cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CqCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

ucs_status_t uct_ugni_destroy_cq(gni_cq_handle_t cq, uct_ugni_cdm_t *cdm)
{
    gni_return_t ugni_rc;

    ugni_rc = GNI_CqDestroy(cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CqDestroy failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}
