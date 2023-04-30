/**
 * Copyright (C) UT-Battelle, LLC. 2022. ALL RIGHTS RESERVED.
 */

#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/debug/log.h>
#include <pthread.h>
#include "ofi_device.h"
static struct fi_info *all = NULL;
static struct fi_info **nics = NULL;
static int num_nics = -1;

/* Info lock to keep mutliple threads from populating the above info all at once */
pthread_mutex_t info_lock = PTHREAD_MUTEX_INITIALIZER;

static ucs_status_t uct_ofi_populate_all()
{
    struct fi_info hints = {0};
    ucs_status_t ret;

    ucs_debug("Populating fi_info structs");
    /* TODO: Maybe FI_FENCE? */
    hints.caps = FI_RMA | FI_ATOMIC | FI_TAGGED;
    hints.addr_format = FI_FORMAT_UNSPEC;

    ret = fi_getinfo(fi_version(), NULL, NULL, 0, &hints, &all);
    if( ret != 0 || !all) {
        ucs_debug("OFI No device was found");
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

ucs_status_t uct_ofi_populate_nics()
{
    ucs_status_t ret = UCS_OK;
    struct fi_info *unique[256];
    int idx = 0, jdx;
    struct fi_info *cur;
    int dup;

    pthread_mutex_lock(&info_lock);

    if (num_nics != -1) {
        goto err_out;
    }
    
    if (!all) {
        ret = uct_ofi_populate_all();
        if (ret != UCS_OK) {
            goto err_out;
        }
    }
    cur = all;
    ucs_debug("populating nic data structs");

    memset(unique, 0, sizeof(struct fi_info*)*256);
    ucs_debug("Starting with fabric %s", cur->domain_attr->name);

    /* Assume fabrics without a nic structure are unintersting */
    while (!cur->nic) {
        ucs_trace("Fabric %s rejected, nic=NULL", cur->domain_attr->name);
        if (!cur->next) {
            ucs_debug("No nic structs found");
            /* reached the end of the list with no nics */
            /* TODO: Maybe instead stop at random TCP/IP address or lo? */
            num_nics = 0;
            ret = UCS_ERR_NO_DEVICE;
            goto err_out;
        }
        cur = cur->next;
    }

    unique[0] = fi_dupinfo(cur);
    cur = cur->next;
    while (cur) {
        for( jdx = 0, dup=0; jdx <= idx; jdx++) {
            if ( !strcmp(cur->domain_attr->name, unique[jdx]->domain_attr->name) ) {
                ucs_trace("OFI device search, rejecting duplicate %s", cur->domain_attr->name);
                dup = 1;
            }
        }
        if( dup ) {
            cur = cur->next;
            continue;
        }
        ucs_trace("Found new name '%s'", cur->domain_attr->name);
        idx++;
        unique[idx] = fi_dupinfo(cur);
        cur = cur->next;
        while (cur && !cur->nic) {
            cur = cur->next;
        }
    }

    num_nics = idx;
    nics = ucs_calloc(idx, sizeof(struct fi_info**), "ofi info pointers");
    memcpy(nics, unique, sizeof(struct fi_info**)*idx);
    ucs_debug("Found %i ofi nics", idx);
err_out:
    pthread_mutex_unlock(&info_lock);
    return ret;
}


ucs_status_t uct_ofi_destroy_fabric(uct_ofi_md_t *md)
{
    int ret;

    ret = fi_close(&md->dom_ctx->fid);
    UCT_OFI_CHECK_ERROR(ret, "Closing domain fid", UCS_ERR_INVALID_PARAM);

    ret = fi_close(&md->fab_ctx->fid);
    UCT_OFI_CHECK_ERROR(ret, "Closing frabric fid", UCS_ERR_INVALID_PARAM);

    fi_freeinfo(all);

    return UCS_OK;
}

/* TODO: more flexible in capabilities */
ucs_status_t uct_ofi_init_fabric(uct_ofi_md_t *md, char *fabric_name)
{
    int ret = 1;

    ucs_trace("Init fabric");

    if (uct_ofi_populate_nics() != UCS_OK){
        return UCS_ERR_NO_DEVICE;
    }

    /* TODO: Make this work so fabrics can be selected by name */
    if( fabric_name ) {
        ucs_error("Selecting dev by name not yet supported");
        return UCS_ERR_NO_DEVICE;
    } else {
        md->fab_info = fi_dupinfo(all);
    }

    /* TODO: version missmatch check here */
    /* Third param is a context for async ops. Could be useful */
    /* TODO: is fab_info needed after this? */
    ret = fi_fabric(md->fab_info->fabric_attr, &md->fab_ctx, NULL);
    UCT_OFI_CHECK_ERROR(ret, "No fabric found", UCS_ERR_NO_DEVICE);

    /* this should be an iface */
    /* or maybe not? */
    ret = fi_domain(md->fab_ctx, md->fab_info, &md->dom_ctx, NULL);
    UCT_OFI_CHECK_ERROR(ret, "Could not make domain", UCS_ERR_NO_DEVICE);
    ucs_debug("Using fabric named: %s", md->fab_info->fabric_attr->name);
    return UCS_OK;
}

static ucs_sys_device_t populate_sys_device(struct fi_info *info)
{
    ucs_sys_device_t sys_dev;
    ucs_sys_bus_id_t bus_id;
    ucs_status_t status;

    if(!info->nic) {
        return UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    bus_id.domain   = info->nic->bus_attr->attr.pci.domain_id;
    bus_id.bus      = info->nic->bus_attr->attr.pci.bus_id;
    bus_id.slot     = info->nic->bus_attr->attr.pci.device_id;
    bus_id.function = info->nic->bus_attr->attr.pci.function_id;

    status = ucs_topo_find_device_by_bus_id(&bus_id, &sys_dev);
    if (status != UCS_OK) {
        return UCS_SYS_DEVICE_ID_UNKNOWN;
    }
    return sys_dev;
}

static void fill_ofi_info(uct_tl_device_resource_t *tl_device,
                          struct fi_info *info)
{
    ucs_snprintf_zero(tl_device->name, sizeof(tl_device->name), "%s",
                      info->domain_attr->name);
    tl_device->type       = UCT_DEVICE_TYPE_NET;
    tl_device->sys_device = populate_sys_device(info);
    ucs_trace("Filled device %s with sys_device %hhx", tl_device->name, tl_device->sys_device);
}

ucs_status_t uct_ofi_query_devices(uct_md_h tl_md,
                                   uct_tl_device_resource_t **tl_devices_p,
                                   unsigned *num_tl_devices_p)
{
    uct_tl_device_resource_t *resources;
    int i;
    ucs_status_t status = UCS_OK;

    ucs_debug("Querying OFI devices");
    status = uct_ofi_populate_nics();
    resources = ucs_calloc(num_nics, sizeof(uct_tl_device_resource_t),
                           "ofi uct_tl_device_resource_t");
    for (i=0; i < num_nics; i++) {
        fill_ofi_info(&resources[i], nics[i]);
    }

 error:
    *num_tl_devices_p = num_nics;
    *tl_devices_p     = resources;

    return status;
}

ucs_status_t uct_ofi_iface_get_dev_address(uct_iface_t *tl_iface, uct_device_addr_t *addr)
{
    return UCS_ERR_UNSUPPORTED;
}
