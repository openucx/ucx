/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "knem_md.h"
#include "knem_io.h"

ucs_status_t uct_knem_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->rkey_packed_size  = sizeof(uct_knem_key_t);
    md_attr->cap.flags         = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_DEFAULT;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->reg_cost.overhead = 1200.0e-9;
    md_attr->reg_cost.growth   = 0.007e-9;

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_knem_query_md_resources(uct_md_resource_desc_t **resources_p,
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
    return uct_single_md_resource(&uct_knem_md_component, resources_p,
                                  num_resources_p);
}

static void uct_knem_md_close(uct_md_h md)
{
    uct_knem_md_t *knem_md = (uct_knem_md_t *)md;
    close(knem_md->knem_fd);
    ucs_free(knem_md);
}

static ucs_status_t uct_knem_mem_reg(uct_md_h md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    int rc;
    struct knem_cmd_create_region create;
    struct knem_cmd_param_iovec knem_iov[1];
    uct_knem_md_t *knem_md = (uct_knem_md_t *)md;
    int knem_fd = knem_md->knem_fd;
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

static ucs_status_t uct_knem_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    int rc;
    uct_knem_key_t *key = (uct_knem_key_t *)memh;
    uct_knem_md_t *knem_md = (uct_knem_md_t *)md;
    int knem_fd = knem_md->knem_fd;

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

static ucs_status_t uct_knem_rkey_pack(uct_md_h md, uct_mem_h memh,
                                       void *rkey_buffer)
{
    uct_knem_key_t *packed = (uct_knem_key_t*)rkey_buffer;
    uct_knem_key_t *key = (uct_knem_key_t *)memh;
    packed->cookie  = (uint64_t)key->cookie;
    packed->address = (uintptr_t)key->address;
    ucs_trace("packed rkey: cookie 0x%"PRIx64" address %"PRIxPTR,
              key->cookie, key->address);
    return UCS_OK;
}

static ucs_status_t uct_knem_rkey_unpack(uct_md_component_t *mdc,
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
    ucs_trace("unpacked rkey: key %p cookie 0x%"PRIx64" address %"PRIxPTR,
              key, key->cookie, key->address);
    return UCS_OK;
}

static ucs_status_t uct_knem_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                          void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t uct_knem_md_open(const char *md_name, const uct_md_config_t *md_config,
                                     uct_md_h *md_p)
{
    uct_knem_md_t *knem_md;

    static uct_md_ops_t md_ops = {
        .close        = uct_knem_md_close,
        .query        = uct_knem_md_query,
        .mem_alloc    = (void*)ucs_empty_function_return_success,
        .mem_free     = (void*)ucs_empty_function_return_success,
        .mkey_pack    = uct_knem_rkey_pack,
        .mem_reg      = uct_knem_mem_reg,
        .mem_dereg    = uct_knem_mem_dereg,
        .mem_type_detect   = ucs_empty_function_return_unsupported,
    };

    knem_md = ucs_malloc(sizeof(uct_knem_md_t), "uct_knem_md_t");
    if (NULL == knem_md) {
        ucs_error("Failed to allocate memory for uct_knem_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    knem_md->super.ops = &md_ops;
    knem_md->super.component = &uct_knem_md_component;

    knem_md->knem_fd = open("/dev/knem", O_RDWR);
    if (knem_md->knem_fd < 0) {
        ucs_error("Could not open the KNEM device file at /dev/knem: %m.");
        free(knem_md);
        return UCS_ERR_IO_ERROR;
    }

    *md_p = (uct_md_h)knem_md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_knem_md_component, "knem",
                        uct_knem_query_md_resources, uct_knem_md_open, 0,
                        uct_knem_rkey_unpack,
                        uct_knem_rkey_release, "KNEM_", uct_md_config_table,
                        uct_md_config_t)
