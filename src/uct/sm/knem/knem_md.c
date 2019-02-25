/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "knem_md.h"
#include "knem_io.h"

#include <ucs/arch/cpu.h>
#include <ucm/api/ucm.h>

static ucs_config_field_t uct_knem_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_knem_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"RCACHE", "try", "Enable using memory registration cache",
     ucs_offsetof(uct_knem_md_config_t, rcache_enable), UCS_CONFIG_TYPE_TERNARY},

    {"", "", NULL,
     ucs_offsetof(uct_knem_md_config_t, rcache),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_rcache_table)},

    {NULL}
};

ucs_status_t uct_knem_md_query(uct_md_h uct_md, uct_md_attr_t *md_attr)
{
    uct_knem_md_t *md = ucs_derived_of(uct_md, uct_knem_md_t);

    md_attr->rkey_packed_size  = sizeof(uct_knem_key_t);
    md_attr->cap.flags         = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_HOST;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->reg_cost          = md->reg_cost;

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
    uct_knem_md_t *knem_md = ucs_derived_of(md, uct_knem_md_t);
    if (knem_md->rcache != NULL) {
        ucs_rcache_destroy(knem_md->rcache);
    }
    close(knem_md->knem_fd);
    ucs_free(knem_md);
}

static ucs_status_t uct_knem_mem_reg_internal(uct_md_h md, void *address, size_t length,
                                              unsigned flags, unsigned silent,
                                              uct_knem_key_t *key)
{
    int rc;
    struct knem_cmd_create_region create;
    struct knem_cmd_param_iovec knem_iov[1];
    uct_knem_md_t *knem_md = (uct_knem_md_t *)md;
    int knem_fd = knem_md->knem_fd;

    ucs_assert_always(knem_fd > -1);

    knem_iov[0].base = (uintptr_t) address;
    knem_iov[0].len = length;

    memset(&create, 0, sizeof(struct knem_cmd_create_region));
    create.iovec_array = (uintptr_t) &knem_iov[0];
    create.iovec_nr = 1;
    create.flags = 0;
    create.protection = PROT_READ | PROT_WRITE;

    rc = ioctl(knem_fd, KNEM_CMD_CREATE_REGION, &create);
    if (rc < 0) {
        if (!silent) {
            /* do not report error in silent mode: it called from rcache
             * internals, rcache will try to register memory again with
             * more accurate data */
            ucs_error("KNEM create region failed: %m");
        }
        return UCS_ERR_IO_ERROR;
    }

    ucs_assert_always(create.cookie != 0);
    key->cookie  = create.cookie;
    key->address = (uintptr_t)address;

    return UCS_OK;
}

static ucs_status_t uct_knem_mem_reg(uct_md_h md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    uct_knem_key_t *key;
    ucs_status_t status;

    key = ucs_malloc(sizeof(uct_knem_key_t), "uct_knem_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_knem_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_knem_mem_reg_internal(md, address, length, flags, 0, key);
    if (status == UCS_OK) {
        *memh_p = key;
    } else {
        ucs_free(key);
    }
     return status;
}

static ucs_status_t uct_knem_mem_dereg_internal(uct_md_h md, uct_knem_key_t *key)
{
    int rc;
    uct_knem_md_t *knem_md = (uct_knem_md_t *)md;
    int knem_fd = knem_md->knem_fd;

    ucs_assert_always(knem_fd > -1);
    ucs_assert_always(key->cookie  != 0);
    ucs_assert_always(key->address != 0);

    rc = ioctl(knem_fd, KNEM_CMD_DESTROY_REGION, &key->cookie);
    if (rc < 0) {
        ucs_error("KNEM destroy region failed, err = %m");
    }

    return UCS_OK;
}

static ucs_status_t uct_knem_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_knem_key_t *key = (uct_knem_key_t *)memh;
    ucs_status_t status;

    status = uct_knem_mem_dereg_internal(md, key);
    if (status == UCS_OK) {
        ucs_free(key);
    }

    return status;
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

static uct_md_ops_t md_ops = {
    .close             = uct_knem_md_close,
    .query             = uct_knem_md_query,
    .mkey_pack         = uct_knem_rkey_pack,
    .mem_reg           = uct_knem_mem_reg,
    .mem_dereg         = uct_knem_mem_dereg,
    .is_mem_type_owned = (void *)ucs_empty_function_return_zero
};

static inline uct_knem_rcache_region_t* uct_knem_rcache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_knem_rcache_region_t, key);
}

static ucs_status_t uct_knem_mem_rcache_reg(uct_md_h uct_md, void *address,
                                            size_t length, unsigned flags,
                                            uct_mem_h *memh_p)
{
    uct_knem_md_t *md = ucs_derived_of(uct_md, uct_knem_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;

    status = ucs_rcache_get(md->rcache, address, length, PROT_READ|PROT_WRITE,
                            &flags, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    *memh_p = &ucs_derived_of(rregion, uct_knem_rcache_region_t)->key;
    return UCS_OK;
}

static ucs_status_t uct_knem_mem_rcache_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_knem_md_t *md                = ucs_derived_of(uct_md, uct_knem_md_t);
    uct_knem_rcache_region_t *region = uct_knem_rcache_region_from_memh(memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t uct_knem_md_rcache_ops = {
    .close             = uct_knem_md_close,
    .query             = uct_knem_md_query,
    .mkey_pack         = uct_knem_rkey_pack,
    .mem_reg           = uct_knem_mem_rcache_reg,
    .mem_dereg         = uct_knem_mem_rcache_dereg,
    .is_mem_type_owned = (void *)ucs_empty_function_return_zero,
};


static ucs_status_t uct_knem_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                               void *arg, ucs_rcache_region_t *rregion,
                                               uint16_t rcache_mem_reg_flags)
{
    uct_knem_rcache_region_t *region = ucs_derived_of(rregion, uct_knem_rcache_region_t);
    uct_knem_md_t *md                = context;
    int *flags                       = arg;

    return uct_knem_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                     region->super.super.end - region->super.super.start,
                                     *flags,
                                     rcache_mem_reg_flags & UCS_RCACHE_MEM_REG_HIDE_ERRORS,
                                     &region->key);
}

static void uct_knem_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *rregion)
{
    uct_knem_rcache_region_t *region = ucs_derived_of(rregion, uct_knem_rcache_region_t);
    uct_knem_md_t            *md     = context;

    uct_knem_mem_dereg_internal(&md->super, &region->key);
}

static void uct_knem_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                           ucs_rcache_region_t *rregion, char *buf,
                                           size_t max)
{
    uct_knem_rcache_region_t *region = ucs_derived_of(rregion, uct_knem_rcache_region_t);
    uct_knem_key_t *key = &region->key;

    snprintf(buf, max, "cookie %"PRIu64" addr %p", key->cookie, (void*)key->address);
}

static ucs_rcache_ops_t uct_knem_rcache_ops = {
    .mem_reg     = uct_knem_rcache_mem_reg_cb,
    .mem_dereg   = uct_knem_rcache_mem_dereg_cb,
    .dump_region = uct_knem_rcache_dump_region_cb
};

static ucs_status_t uct_knem_md_open(const char *md_name,
                                     const uct_md_config_t *uct_md_config,
                                     uct_md_h *md_p)
{
    const uct_knem_md_config_t *md_config = ucs_derived_of(uct_md_config, uct_knem_md_config_t);
    uct_knem_md_t *knem_md;
    ucs_rcache_params_t rcache_params;
    ucs_status_t status;

    knem_md = ucs_malloc(sizeof(uct_knem_md_t), "uct_knem_md_t");
    if (NULL == knem_md) {
        ucs_error("Failed to allocate memory for uct_knem_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    knem_md->super.ops         = &md_ops;
    knem_md->super.component   = &uct_knem_md_component;
    knem_md->reg_cost.overhead = 1200.0e-9;
    knem_md->reg_cost.growth   = 0.007e-9;
    knem_md->rcache            = NULL;

    knem_md->knem_fd = open("/dev/knem", O_RDWR);
    if (knem_md->knem_fd < 0) {
        ucs_error("Could not open the KNEM device file at /dev/knem: %m.");
        free(knem_md);
        return UCS_ERR_IO_ERROR;
    }

    if (md_config->rcache_enable != UCS_NO) {
        rcache_params.region_struct_size = sizeof(uct_knem_rcache_region_t);
        rcache_params.alignment          = md_config->rcache.alignment;
        rcache_params.max_alignment      = ucs_get_page_size();
        rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED;
        rcache_params.ucm_event_priority = md_config->rcache.event_prio;
        rcache_params.context            = knem_md;
        rcache_params.ops                = &uct_knem_rcache_ops;
        status = ucs_rcache_create(&rcache_params, "knem rcache device",
                                   ucs_stats_get_root(), &knem_md->rcache);
        if (status == UCS_OK) {
            knem_md->super.ops         = &uct_knem_md_rcache_ops;
            knem_md->reg_cost.overhead = md_config->rcache.overhead;
            knem_md->reg_cost.growth   = 0; /* It's close enough to 0 */
        } else {
            ucs_assert(knem_md->rcache == NULL);
            if (md_config->rcache_enable == UCS_YES) {
                ucs_error("Failed to create registration cache: %s",
                          ucs_status_string(status));
                uct_knem_md_close(&knem_md->super);
                return status;
            } else {
                ucs_debug("Could not create registration cache: %s",
                          ucs_status_string(status));
            }
        }
    }

    *md_p = (uct_md_h)knem_md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_knem_md_component, "knem",
                        uct_knem_query_md_resources, uct_knem_md_open, 0,
                        uct_knem_rkey_unpack,
                        uct_knem_rkey_release, "KNEM_",
                        uct_knem_md_config_table, uct_knem_md_config_t,
                        ucs_empty_function_return_unsupported)
