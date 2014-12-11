/**
 * Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#define _GNU_SOURCE /* for CPU_ZERO/CPU_SET in sched.h */
#include "ugni_device.h"

#include <uct/tl/context.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


#define UCT_UGNI_RKEY_MAGIC  0x77777777ul

static ucs_status_t uct_ugni_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->rkey_packed_size  = sizeof(uint64_t);
    return UCS_OK;
}

static ucs_status_t uct_ugni_mem_map(uct_pd_h pd, void **address_p, size_t *length_p,
                                     unsigned flags, uct_lkey_t *lkey_p UCS_MEMTRACK_ARG)
{
#if 0
    uct_ib_device_t *dev = ucs_derived_of(pd, uct_ib_device_t);
    struct ibv_mr *mr;

    mr = ibv_reg_mr(dev->pd, address, length,
            IBV_ACCESS_LOCAL_WRITE |
            IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ |
            IBV_ACCESS_REMOTE_ATOMIC);
    if (mr == NULL) {
        ucs_error("ibv_reg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    *lkey_p = (uintptr_t)mr;
#endif
    return UCS_OK;
}

static ucs_status_t uct_ugni_mem_unmap(uct_pd_h pd, uct_lkey_t lkey)
{
#if 0
    struct ibv_mr *mr = (void*)lkey;
    int ret;

    ret = ibv_dereg_mr(mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

#endif
    return UCS_OK;
}

static ucs_status_t uct_ugni_rkey_pack(uct_pd_h pd, uct_lkey_t lkey,
        void *rkey_buffer)
{
#if 0
    struct ibv_mr *mr = (void*)lkey;
    uint32_t *ptr = rkey_buffer;

    *(ptr++) = UCT_IB_RKEY_MAGIC;
    *(ptr++) = htonl(mr->rkey); /* Use r-keys as big endian */
#endif
    return UCS_OK;
}

ucs_status_t uct_ugni_rkey_unpack(uct_context_h context, void *rkey_buffer,
        uct_rkey_bundle_t *rkey_ob)
{
#if 0
    uint32_t *ptr = rkey_buffer;
    uint32_t magic;

    magic = *(ptr++);
    if (magic != UCT_IB_RKEY_MAGIC) {
        return UCS_ERR_UNSUPPORTED;
    }

    rkey_ob->rkey = *(ptr++);
    rkey_ob->type = (void*)ucs_empty_function;
#endif
    return UCS_OK;
}

void uct_device_get_resource(uct_ugni_device_t *dev,
        uct_resource_desc_t *resource)
{
    ucs_snprintf_zero(resource->tl_name,  sizeof(resource->tl_name), "%s", TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s", dev->fname);
    resource->addrlen    = sizeof(unsigned int);
    resource->local_cpus = dev->cpu_mask;
    resource->latency    = 900; /* nano sec*/
    resource->bandwidth  = (long) (6911 * pow(1024,2));
    memset(&resource->subnet_addr, 0, sizeof(resource->subnet_addr));
}

uct_pd_ops_t uct_ugni_pd_ops = {
    .query        = uct_ugni_pd_query,
    .mem_map      = uct_ugni_mem_map,
    .mem_unmap    = uct_ugni_mem_unmap,
    .rkey_pack    = uct_ugni_rkey_pack,
};

ucs_status_t uct_ugni_device_create(uct_context_h context, int dev_id, uct_ugni_device_t *dev_p)
{
    gni_return_t ugni_rc;

    dev_p->device_id = (uint32_t)dev_id;

    ugni_rc = GNI_CdmGetNicAddress(dev_p->device_id, &dev_p->address,
                                   &dev_p->cpu_id);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmGetNicAddress failed, device %d, Error status: %s %d",
                  dev_id, gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }
    CPU_SET(dev_p->cpu_id, &(dev_p->cpu_mask));

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

    ucs_snprintf_zero(dev_p->fname, sizeof(dev_p->fname), "%s:%u",
                      dev_p->type_name, dev_p->address);

    dev_p->super.ops = &uct_ugni_pd_ops;
    dev_p->super.context = context;
    dev_p->attached = false;
    return UCS_OK;
}

void uct_ugni_device_destroy(uct_ugni_device_t *dev)
{
    /* Nop */
}
