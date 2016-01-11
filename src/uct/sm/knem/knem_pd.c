/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "knem_pd.h"
#include "knem_io.h"

ucs_status_t uct_knem_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->rkey_packed_size  = sizeof(uct_knem_key_t);
    pd_attr->cap.flags         = UCT_PD_FLAG_REG;
    pd_attr->cap.max_alloc     = 0;
    pd_attr->cap.max_reg       = ULONG_MAX;
    pd_attr->reg_cost.overhead = 1000.0e-9;
    pd_attr->reg_cost.growth   = 0.007e-9;

    memset(&pd_attr->local_cpus, 0xff, sizeof(pd_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_knem_query_pd_resources(uct_pd_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    int fd;
    int rc;
    struct knem_cmd_info info;

    memset(&info, 0, sizeof(struct knem_cmd_info));

    fd = open("/dev/knem", O_RDWR);
    if (fd < 0) {
        ucs_debug("Could not open the KNEM device file at /dev/knem: %m. Disabling knem resource");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    rc = ioctl(fd, KNEM_CMD_GET_INFO, &info);
    if (rc < 0) {
        *resources_p     = NULL;
        *num_resources_p = 0;
        close(fd);
        ucs_debug("KNEM get info failed. not using knem, err = %d %m", rc);
        return UCS_OK;
    }

    if (KNEM_ABI_VERSION != info.abi) {
        *resources_p     = NULL;
        *num_resources_p = 0;
        close(fd);
        ucs_error("KNEM ABI mismatch: KNEM_ABI_VERSION: %d, Driver binary interface version: %d",
                  KNEM_ABI_VERSION, info.abi);
        return UCS_OK;
    }

    /* We have to close it since it is not clear
     * if it will be selected in future */
    close(fd);
    return uct_single_pd_resource(&uct_knem_pd_component, resources_p,
                                  num_resources_p);
}

static void uct_knem_pd_close(uct_pd_h pd)
{
    uct_knem_pd_t *knem_pd = (uct_knem_pd_t *)pd;
    close(knem_pd->knem_fd);
    ucs_free(knem_pd);
}

static ucs_status_t uct_knem_mem_reg(uct_pd_h pd, void *address, size_t length,
                                     uct_mem_h *memh_p)
{
    int rc;
    struct knem_cmd_create_region create;
    struct knem_cmd_param_iovec knem_iov[1];
    uct_knem_pd_t *knem_pd = (uct_knem_pd_t *)pd;
    int knem_fd = knem_pd->knem_fd;
    uct_knem_key_t *key;

    ucs_assert_always(knem_fd > -1);

    key = ucs_malloc(sizeof(uct_knem_key_t), "uct_knem_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_knem_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    knem_iov[0].base = (uintptr_t) address;
    knem_iov[0].len = length;

    memset(&create, 0, sizeof(struct knem_cmd_create_region));
    create.iovec_array = (uintptr_t) &knem_iov[0];
    create.iovec_nr = 1;
    create.flags = 0;
    create.protection = PROT_READ | PROT_WRITE;

    rc = ioctl(knem_fd, KNEM_CMD_CREATE_REGION, &create);
    if (rc < 0) {
        ucs_error("KNEM create region failed: %m");
        ucs_free(key);
        return UCS_ERR_IO_ERROR;
    }

    ucs_assert_always(create.cookie != 0);
    key->cookie  = create.cookie;
    key->address = (uintptr_t)address;

    *memh_p = key;
    return UCS_OK;
}

static ucs_status_t uct_knem_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    int rc;
    uct_knem_key_t *key = (uct_knem_key_t *)memh;
    uct_knem_pd_t *knem_pd = (uct_knem_pd_t *)pd;
    int knem_fd = knem_pd->knem_fd;

    ucs_assert_always(knem_fd > -1);
    ucs_assert_always(key->cookie  != 0);
    ucs_assert_always(key->address != 0);

    rc = ioctl(knem_fd, KNEM_CMD_DESTROY_REGION, &key->cookie);
    if (rc < 0) {
        ucs_error("KNEM destroy region failed, err = %m");
    }

    ucs_free(key);
    return UCS_OK;
}

static ucs_status_t uct_knem_rkey_pack(uct_pd_h pd, uct_mem_h memh,
                                       void *rkey_buffer)
{
    uct_knem_key_t *packed = (uct_knem_key_t*)rkey_buffer;
    uct_knem_key_t *key = (uct_knem_key_t *)memh;
    packed->cookie  = (uint64_t)key->cookie;
    packed->address = (uintptr_t)key->address;
    ucs_trace("packed rkey: cookie %"PRIu64" address %"PRIxPTR,
              key->cookie, key->address);
    return UCS_OK;
}

static ucs_status_t uct_knem_rkey_unpack(uct_pd_component_t *pdc,
                                         const void *rkey_buffer, uct_rkey_t *rkey_p,
                                         void **handle_p)
{
    uct_knem_key_t *packed = (uct_knem_key_t *)rkey_buffer;
    uct_knem_key_t *key;

    key = ucs_malloc(sizeof(uct_knem_key_t), "uct_knem_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_knem_key_t");
        return UCS_ERR_NO_MEMORY;
    }
    key->cookie = packed->cookie;
    key->address = packed->address;
    *handle_p = NULL;
    *rkey_p = (uintptr_t)key;
    ucs_trace("unpacked rkey: key %p cookie %"PRIu64" address %"PRIxPTR,
              key, key->cookie, key->address);
    return UCS_OK;
}

static ucs_status_t uct_knem_rkey_release(uct_pd_component_t *pdc, uct_rkey_t rkey,
                                          void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t uct_knem_pd_open(const char *pd_name, const uct_pd_config_t *pd_config,
                                     uct_pd_h *pd_p)
{
    uct_knem_pd_t *knem_pd;

    static uct_pd_ops_t pd_ops = {
        .close        = uct_knem_pd_close,
        .query        = uct_knem_pd_query,
        .mem_alloc    = (void*)ucs_empty_function_return_success,
        .mem_free     = (void*)ucs_empty_function_return_success,
        .mkey_pack    = uct_knem_rkey_pack,
        .mem_reg      = uct_knem_mem_reg,
        .mem_dereg    = uct_knem_mem_dereg
    };

    knem_pd = ucs_malloc(sizeof(uct_knem_pd_t), "uct_knem_pd_t");
    if (NULL == knem_pd) {
        ucs_error("Failed to allocate memory for uct_knem_pd_t");
        return UCS_ERR_NO_MEMORY;
    }

    knem_pd->super.ops = &pd_ops;
    knem_pd->super.component = &uct_knem_pd_component;

    knem_pd->knem_fd = open("/dev/knem", O_RDWR);
    if (knem_pd->knem_fd < 0) {
        ucs_error("Could not open the KNEM device file at /dev/knem: %m.");
        free(knem_pd);
        return UCS_ERR_IO_ERROR;
    }

    *pd_p = (uct_pd_h)knem_pd;
    return UCS_OK;
}

UCT_PD_COMPONENT_DEFINE(uct_knem_pd_component, "knem",
                        uct_knem_query_pd_resources, uct_knem_pd_open, 0,
                        uct_knem_rkey_unpack,
                        uct_knem_rkey_release, "KNEM_", uct_pd_config_table,
                        uct_pd_config_t)
