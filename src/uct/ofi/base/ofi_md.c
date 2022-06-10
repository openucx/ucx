/**
 * Copyright (C) UT-Battelle, LLC. 2022. ALL RIGHTS
 */

#include "ofi_md.h"
#include "ofi_def.h"
#include <ucs/debug/log.h>
#include <stdint.h>

/* TODO: Looks like RKEYs maynot be needed, need to adjust since right now I'm just assuming it does */
/* FI_PROGRESS_UNSPEC */




static ucs_status_t uct_ofi_query_md_resources(uct_component_h component,
                           uct_md_resource_desc_t **resources_p,
                           unsigned *num_resources_p)
{
    uct_md_resource_desc_t *resource;

    ucs_trace("OFI query resources");
    resource = ucs_malloc(sizeof(*resource)*1, "md resource");
    if (resource == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* TODO: Query if hmem is available and make it selectable. Or is this
     * the wrong approach?  */
    ucs_snprintf_zero(resource[0].md_name, UCT_MD_NAME_MAX, "%s sys memory",
                      component->name);

    *resources_p     = resource;
    *num_resources_p = 1;
    return UCS_OK;
}

static ucs_status_t uct_ofi_md_query(uct_md_h tl_md, uct_md_attr_t *md_attr)
{
    uct_ofi_md_t *md = ucs_derived_of(tl_md, uct_ofi_md_t);

    ucs_trace("OFI md query");
    md_attr->cap.flags            = UCT_MD_FLAG_REG;
    /* TODO: hmem option */
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.alloc_mem_types  = 0;
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    /* TODO: Can libfabric detect? */
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;
    /* TODO: find correct measure */
    md_attr->reg_cost             = ucs_linear_func_make(1000.0e-9, 0.007e-9);
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    md_attr->rkey_packed_size     = md->fab_info->domain_attr->mr_key_size;
    return UCS_OK;
}


static void uct_ofi_md_close(uct_md_h md)
{
    uct_ofi_md_t *ofi_md = ucs_derived_of(md, uct_ofi_md_t);
    ucs_trace("OFI md close");

    ofi_md->ref_count--;
    if (!ofi_md->ref_count) {
        ucs_debug("Tearing down OFI domain");
        
	/* TODO: teardown code */
    }
}


static ucs_status_t uct_ofi_mem_reg(uct_md_h md, void *address, size_t length,
                                    unsigned flags, uct_mem_h *memh_p)
{
    uct_ofi_md_t *ofi_md = ucs_derived_of(md, uct_ofi_md_t);
    int ret;
    struct fid_mr **mr = (struct fid_mr **)memh_p;

    ucs_debug("address=%p length=%zd", address, length);

    ret = fi_mr_reg(ofi_md->dom_ctx, address, length, FI_REMOTE_WRITE | FI_REMOTE_READ | FI_READ | FI_WRITE,
                    0, 0ULL, 0, mr, NULL);

    UCT_OFI_CHECK_ERROR(ret, "fi_mr_reg", UCS_ERR_NO_MEMORY);
    return UCS_OK;
}


static ucs_status_t uct_ofi_mem_dereg(uct_md_h md,
                                      const uct_md_mem_dereg_params_t *params)
{
    int ret;
    struct fid_mr *mr = (struct fid_mr *)params->memh;
    ucs_trace("uct_ofi_mem_dereg");
    if (params->field_mask & UCT_MD_MEM_DEREG_FIELD_FLAGS && params->flags & UCT_MD_MEM_DEREG_FLAG_INVALIDATE) {
        return UCS_ERR_UNSUPPORTED;
    }
    ret = fi_close(&mr->fid);
    UCT_OFI_CHECK_ERROR(ret, "fi_close on mem dereg", UCS_ERR_IO_ERROR);
    return UCS_OK;
}


static ucs_status_t uct_ofi_rkey_pack(uct_md_h md, uct_mem_h memh,
                                      void *rkey_buffer)
{
    return UCS_ERR_NO_MEMORY;
}


static ucs_status_t uct_ofi_rkey_release(uct_component_t *component,
                                         uct_rkey_t rkey, void *handle)
{
    return UCS_ERR_IO_ERROR;
}


static ucs_status_t uct_ofi_rkey_unpack(uct_component_t *component,
                                        const void *rkey_buffer,
                                        uct_rkey_t *rkey_p, void **handle_p)
{
    return UCS_ERR_UNSUPPORTED;
}


static ucs_status_t
uct_ofi_md_open(uct_component_h component, const char *md_name,
                 const uct_md_config_t *md_config, uct_md_h *md_p)
{
    ucs_status_t status = UCS_OK;
    static uct_md_ops_t md_ops;
    static uct_ofi_md_t md;

    md_ops.close              = uct_ofi_md_close;
    md_ops.query              = uct_ofi_md_query;
    md_ops.mem_alloc          = (void*)ucs_empty_function;
    md_ops.mem_free           = (void*)ucs_empty_function;
    md_ops.mem_reg            = uct_ofi_mem_reg;
    md_ops.mem_dereg          = uct_ofi_mem_dereg;
    md_ops.mkey_pack          = uct_ofi_rkey_pack;
    md_ops.detect_memory_type = ucs_empty_function_return_unsupported;
    md_ops.is_sockaddr_accessible = ucs_empty_function_return_zero_int;

    md.super.ops              = &md_ops;
    md.super.component        = &uct_ofi_component;
    md.ref_count              = 0;

    *md_p = &md.super;

    ucs_trace("OFI MD open");

    if (!md.ref_count) {
        /* TODO: Logic to let users request fabrics on the command line */
        /* Hard coding NULL takes the first available domain. Can this be done smarter? */
        status = uct_ofi_init_fabric(&md, NULL);
        if (UCS_OK != status) {
	    goto error;
        } 
    }

    md.ref_count++;

error:
    return status;
}


uct_component_t uct_ofi_component = {
    .query_md_resources = uct_ofi_query_md_resources,
    .md_open            = uct_ofi_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ofi_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_ofi_rkey_release,
    .name               = UCT_OFI_MD_NAME,
    .md_config          = {
        .name           = "Libfabric memory domain",
        .prefix         = "FI_",
	.table          = uct_md_config_table,
        .size           = sizeof(uct_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_ofi_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_ofi_component);
