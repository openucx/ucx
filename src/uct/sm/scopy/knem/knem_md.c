/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "knem_md.h"
#include "knem_io.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucm/api/ucm.h>
#include <ucs/vfs/base/vfs_obj.h>


#define UCT_KNEM_MD_MEM_DEREG_CHECK_PARAMS(_params) \
    UCT_MD_MEM_DEREG_CHECK_PARAMS(_params, 0)

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

    md_attr->rkey_packed_size     = sizeof(uct_knem_key_t);
    md_attr->cap.flags            = UCT_MD_FLAG_REG |
                                    UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.alloc_mem_types  = 0;
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;
    md_attr->reg_cost             = md->reg_cost;

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t
uct_knem_query_md_resources(uct_component_t *component,
                            uct_md_resource_desc_t **resources_p,
                            unsigned *num_resources_p)
{
    int fd;
    int rc;
    struct knem_cmd_info info;

    memset(&info, 0, sizeof(struct knem_cmd_info));

    fd = open("/dev/knem", O_RDWR);
    if (fd < 0) {
        ucs_debug("could not open the KNEM device file at /dev/knem: %m. Disabling knem resource");
        goto out_empty;
    }

    rc = ioctl(fd, KNEM_CMD_GET_INFO, &info);
    if (rc < 0) {
        ucs_debug("KNEM get info failed. not using knem, err = %d %m", rc);
        goto out_empty_close_fd;
    }

    if (KNEM_ABI_VERSION != info.abi) {
        ucs_error("KNEM ABI mismatch: KNEM_ABI_VERSION: %d, Driver binary interface version: %d",
                  KNEM_ABI_VERSION, info.abi);
        goto out_empty_close_fd;
    }

    /* We have to close it since it is not clear
     * if it will be selected in future */
    close(fd);
    return uct_md_query_single_md_resource(component, resources_p, num_resources_p);

out_empty_close_fd:
    close(fd);
out_empty:
    return uct_md_query_empty_md_resource(resources_p, num_resources_p);
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
        if (!silent && !(flags & UCT_MD_MEM_FLAG_HIDE_ERRORS)) {
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

static ucs_status_t uct_knem_mem_dereg(uct_md_h md,
                                       const uct_md_mem_dereg_params_t *params)
{
    uct_knem_key_t *key;
    ucs_status_t status;

    UCT_KNEM_MD_MEM_DEREG_CHECK_PARAMS(params);

    key    = (uct_knem_key_t *)params->memh;
    status = uct_knem_mem_dereg_internal(md, key);
    if (status == UCS_OK) {
        ucs_free(key);
    }

    return status;
}

static ucs_status_t
uct_knem_rkey_pack(uct_md_h md, uct_mem_h memh,
                   const uct_md_mkey_pack_params_t *params,
                   void *rkey_buffer)
{
    uct_knem_key_t *packed = rkey_buffer;
    uct_knem_key_t *key    = memh;

    packed->cookie  = (uint64_t)key->cookie;
    packed->address = (uintptr_t)key->address;
    ucs_trace("packed rkey: cookie 0x%"PRIx64" address %"PRIxPTR,
              key->cookie, key->address);
    return UCS_OK;
}

static ucs_status_t uct_knem_rkey_unpack(uct_component_t *component,
                                         const void *rkey_buffer,
                                         uct_rkey_t *rkey_p, void **handle_p)
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

static ucs_status_t uct_knem_rkey_release(uct_component_t *component,
                                          uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static uct_md_ops_t md_ops = {
    .close              = uct_knem_md_close,
    .query              = uct_knem_md_query,
    .mkey_pack          = uct_knem_rkey_pack,
    .mem_reg            = uct_knem_mem_reg,
    .mem_dereg          = uct_knem_mem_dereg,
    .detect_memory_type = ucs_empty_function_return_unsupported,
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

static ucs_status_t
uct_knem_mem_rcache_dereg(uct_md_h uct_md,
                          const uct_md_mem_dereg_params_t *params)
{
    uct_knem_md_t *md = ucs_derived_of(uct_md, uct_knem_md_t);
    uct_knem_rcache_region_t *region;
 
    UCT_KNEM_MD_MEM_DEREG_CHECK_PARAMS(params);
 
    region = uct_knem_rcache_region_from_memh(params->memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t uct_knem_md_rcache_ops = {
    .close                  = uct_knem_md_close,
    .query                  = uct_knem_md_query,
    .mkey_pack              = uct_knem_rkey_pack,
    .mem_reg                = uct_knem_mem_rcache_reg,
    .mem_dereg              = uct_knem_mem_rcache_dereg,
    .is_sockaddr_accessible = ucs_empty_function_return_zero_int,
    .detect_memory_type     = ucs_empty_function_return_unsupported,
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

static ucs_status_t
uct_knem_md_open(uct_component_t *component, const char *md_name,
                 const uct_md_config_t *uct_md_config, uct_md_h *md_p)
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

    knem_md->super.ops       = &md_ops;
    knem_md->super.component = &uct_knem_component;
    knem_md->reg_cost        = ucs_linear_func_make(1200.0e-9, 0.007e-9);
    knem_md->rcache          = NULL;

    knem_md->knem_fd = open("/dev/knem", O_RDWR);
    if (knem_md->knem_fd < 0) {
        ucs_error("Could not open the KNEM device file at /dev/knem: %m.");
        ucs_free(knem_md);
        return UCS_ERR_IO_ERROR;
    }

    if (md_config->rcache_enable != UCS_NO) {
        uct_md_set_rcache_params(&rcache_params, &md_config->rcache);
        rcache_params.region_struct_size = sizeof(uct_knem_rcache_region_t);
        rcache_params.max_alignment      = ucs_get_page_size();
        rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED;
        rcache_params.context            = knem_md;
        rcache_params.ops                = &uct_knem_rcache_ops;
        status = ucs_rcache_create(&rcache_params, "knem", ucs_stats_get_root(),
                                   &knem_md->rcache);
        if (status == UCS_OK) {
            knem_md->super.ops = &uct_knem_md_rcache_ops;
            knem_md->reg_cost  = ucs_linear_func_make(
                                 uct_md_rcache_overhead(&md_config->rcache), 0);
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

static void uct_knem_md_vfs_init(uct_md_h md)
{
    uct_knem_md_t *knem_md = (uct_knem_md_t*)md;

    if (knem_md->rcache != NULL) {
        ucs_vfs_obj_add_sym_link(md, knem_md->rcache, "rcache");
    }
}

uct_component_t uct_knem_component = {
    .query_md_resources = uct_knem_query_md_resources,
    .md_open            = uct_knem_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_knem_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_knem_rkey_release,
    .name               = "knem",
    .md_config          = {
        .name           = "KNEM memory domain",
        .prefix         = "KNEM_",
        .table          = uct_knem_md_config_table,
        .size           = sizeof(uct_knem_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_knem_component),
    .flags              = 0,
    .md_vfs_init        = uct_knem_md_vfs_init
};
