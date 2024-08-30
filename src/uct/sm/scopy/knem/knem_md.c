/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
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

    {NULL}
};

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
            ucs_error("KNEM failed to create region address %p length %zi: %m",
                      address, length);
        }
        return UCS_ERR_IO_ERROR;
    }

    ucs_assert_always(create.cookie != 0);
    key->cookie  = create.cookie;
    key->address = (uintptr_t)address;

    return UCS_OK;
}

static ucs_status_t uct_knem_mem_reg(uct_md_h md, void *address, size_t length,
                                     const uct_md_mem_reg_params_t *params,
                                     uct_mem_h *memh_p)
{
    uint64_t flags = UCT_MD_MEM_REG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
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

static void uct_knem_mem_dereg_internal(uct_md_h md, uct_knem_key_t *key)
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
}

static ucs_status_t uct_knem_mem_dereg(uct_md_h md,
                                       const uct_md_mem_dereg_params_t *params)
{
    uct_knem_key_t *key;

    UCT_KNEM_MD_MEM_DEREG_CHECK_PARAMS(params);

    key = (uct_knem_key_t*)params->memh;
    uct_knem_mem_dereg_internal(md, key);
    ucs_free(key);

    return UCS_OK;
}

int uct_knem_md_check_mem_reg(uct_md_h md)
{
    uint8_t buff;
    uct_knem_key_t key;

    if (uct_knem_mem_reg_internal(md, &buff, sizeof(buff), 0, 1, &key) !=
        UCS_OK) {
        return 0;
    }

    uct_knem_mem_dereg_internal(md, &key);
    return 1;
}

ucs_status_t uct_knem_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr)
{
    uct_knem_md_t *md = ucs_derived_of(uct_md, uct_knem_md_t);

    uct_md_base_md_query(md_attr);
    md_attr->flags                  = UCT_MD_FLAG_NEED_RKEY;
    if (uct_knem_md_check_mem_reg(uct_md)) {
        md_attr->flags         |= UCT_MD_FLAG_REG;
        md_attr->reg_mem_types |= UCS_BIT(UCS_MEMORY_TYPE_HOST);
    }

    md_attr->rkey_packed_size = sizeof(uct_knem_key_t);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->reg_cost         = md->reg_cost;
    return UCS_OK;
}

static ucs_status_t
uct_knem_mkey_pack(uct_md_h md, uct_mem_h memh, void *address, size_t length,
                   const uct_md_mkey_pack_params_t *params,
                   void *mkey_buffer)
{
    uct_knem_key_t *packed = mkey_buffer;
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
    .mkey_pack          = uct_knem_mkey_pack,
    .mem_reg            = uct_knem_mem_reg,
    .mem_dereg          = uct_knem_mem_dereg,
    .mem_attach         = ucs_empty_function_return_unsupported,
    .detect_memory_type = ucs_empty_function_return_unsupported,
};

static ucs_status_t
uct_knem_md_open(uct_component_t *component, const char *md_name,
                 const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    uct_knem_md_t *knem_md;

    knem_md = ucs_malloc(sizeof(uct_knem_md_t), "uct_knem_md_t");
    if (NULL == knem_md) {
        ucs_error("Failed to allocate memory for uct_knem_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    knem_md->super.ops       = &md_ops;
    knem_md->super.component = &uct_knem_component;
    knem_md->reg_cost        = ucs_linear_func_make(1200.0e-9, 0.007e-9);

    knem_md->knem_fd = open("/dev/knem", O_RDWR);
    if (knem_md->knem_fd < 0) {
        ucs_error("Could not open the KNEM device file at /dev/knem: %m.");
        ucs_free(knem_md);
        return UCS_ERR_IO_ERROR;
    }

    *md_p = (uct_md_h)knem_md;
    return UCS_OK;
}

uct_component_t uct_knem_component = {
    .query_md_resources = uct_knem_query_md_resources,
    .md_open            = uct_knem_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_knem_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_knem_rkey_release,
    .rkey_compare       = uct_base_rkey_compare,
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
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
