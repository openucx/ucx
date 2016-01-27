/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "mm_pd.h"

ucs_config_field_t uct_mm_pd_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_mm_pd_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_pd_config_table)},

  {"HUGETLB_MODE", "yes",
   "Enable using huge pages for internal buffers. "
   "Possible values are:\n"
   " y   - Allocate memory using huge pages only.\n"
   " n   - Allocate memory using regular pages only.\n"
   " try - Try to allocate memory using huge pages and if it fails, allocate regular pages.\n",
   ucs_offsetof(uct_mm_pd_config_t, hugetlb_mode), UCS_CONFIG_TYPE_TERNARY},

  {NULL}
};

ucs_status_t uct_mm_mem_alloc(uct_pd_h pd, size_t *length_p, void **address_p,
                              uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    seg = ucs_calloc(1, sizeof(*seg), "mm_seg");
    if (NULL == seg) {
        ucs_error("Failed to allocate memory for mm segment");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_mm_pd_mapper_ops(pd)->alloc(pd, length_p, UCS_TRY, &seg->address,
                                             &seg->mmid, &seg->path
                                             UCS_MEMTRACK_VAL);
    if (status != UCS_OK) {
        ucs_free(seg);
        return status;
    }

    seg->length = *length_p;
    *address_p  = seg->address;
    *memh_p     = seg;

    ucs_debug("mm allocated address %p length %zu mmid %"PRIu64,
              *address_p, seg->length, seg->mmid);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_free(uct_pd_h pd, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = uct_mm_pd_mapper_ops(pd)->free(seg->address, seg->mmid, seg->length,
                                            seg->path);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(seg);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_reg(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    seg = ucs_malloc(sizeof(*seg), "mm_seg");
    if (NULL == seg) {
        ucs_error("Failed to allocate memory for mm segment");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_mm_pd_mapper_ops(pd)->reg(address, length, 
                                           &seg->mmid);
    if (status != UCS_OK) {
        ucs_free(seg);
        return status;
    }

    seg->length  = length;
    seg->address = address;
    *memh_p      = seg;

    ucs_debug("mm registered address %p length %zu mmid %"PRIu64,
              address, length, seg->mmid);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = uct_mm_pd_mapper_ops(pd)->dereg(seg->mmid);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(seg);
    return UCS_OK;
}

ucs_status_t uct_mm_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->cap.flags     = 0;
    if (uct_mm_pd_mapper_ops(pd)->alloc != NULL) {
        pd_attr->cap.flags |= UCT_PD_FLAG_ALLOC;
    }
    if (uct_mm_pd_mapper_ops(pd)->reg != NULL) {
        pd_attr->cap.flags |= UCT_PD_FLAG_REG;
        pd_attr->reg_cost.overhead = 1000.0e-9;
        pd_attr->reg_cost.growth   = 0.007e-9;
    }
    pd_attr->cap.max_alloc    = ULONG_MAX;
    pd_attr->cap.max_reg      = 0;
    pd_attr->rkey_packed_size = sizeof(uct_mm_packed_rkey_t) +
                                uct_mm_pd_mapper_ops(pd)->get_path_size(pd);
    memset(&pd_attr->local_cpus, 0xff, sizeof(pd_attr->local_cpus));
    return UCS_OK;
}

ucs_status_t uct_mm_mkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer)
{
    uct_mm_packed_rkey_t *rkey = rkey_buffer;
    uct_mm_seg_t *seg = memh;

    rkey->mmid      = seg->mmid;
    rkey->owner_ptr = (uintptr_t)seg->address;
    rkey->length    = seg->length;

    if (seg->path != NULL) {
        strcpy(rkey->path, seg->path);
    }

    ucs_trace("packed rkey: mmid %"PRIu64" owner_ptr %"PRIxPTR,
              rkey->mmid, rkey->owner_ptr);
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_unpack(uct_pd_component_t *pdc, const void *rkey_buffer,
                                uct_rkey_t *rkey_p, void **handle_p)
{
    /* user is responsible to free rkey_buffer */
    const uct_mm_packed_rkey_t *rkey = rkey_buffer;
    uct_mm_remote_seg_t *mm_desc;
    ucs_status_t status;

    ucs_trace("unpacking rkey: mmid %"PRIu64" owner_ptr %"PRIxPTR,
              rkey->mmid, rkey->owner_ptr);

    mm_desc = ucs_malloc(sizeof(*mm_desc), "mm_desc");
    if (mm_desc == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    status = uct_mm_pdc_mapper_ops(pdc)->attach(rkey->mmid, rkey->length,
                                                (void *)rkey->owner_ptr, 
                                                &mm_desc->address,
                                                &mm_desc->cookie,
                                                rkey->path);
    if (status != UCS_OK) {
        ucs_free(mm_desc);
        return status;
    }

    mm_desc->length = rkey->length;
    mm_desc->mmid   = rkey->mmid;
    /* store the offset of the addresses, this can be used directly to translate
     * the remote VA to local VA of the attached segment */
    *handle_p = mm_desc;
    *rkey_p   = (uintptr_t)mm_desc->address - rkey->owner_ptr;
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_release(uct_pd_component_t *pdc, uct_rkey_t rkey, void *handle)
{
    ucs_status_t status;
    uct_mm_remote_seg_t *mm_desc = handle;

    status = uct_mm_pdc_mapper_ops(pdc)->detach(mm_desc);
    ucs_free(mm_desc);
    return status;
}

static void uct_mm_pd_close(uct_pd_h pd)
{
    uct_mm_pd_t *mm_pd = ucs_derived_of(pd, uct_mm_pd_t);

    ucs_config_parser_release_opts(mm_pd->config, pd->component->pd_config_table);
    ucs_free(mm_pd->config);
    ucs_free(mm_pd);
}

uct_pd_ops_t uct_mm_pd_ops = {
    .close        = uct_mm_pd_close,
    .query        = uct_mm_pd_query,
    .mem_alloc    = uct_mm_mem_alloc,
    .mem_free     = uct_mm_mem_free,
    .mem_reg      = uct_mm_mem_reg,
    .mem_dereg    = uct_mm_mem_dereg,
    .mkey_pack    = uct_mm_mkey_pack,
};

ucs_status_t uct_mm_pd_open(const char *pd_name, const uct_pd_config_t *pd_config,
                            uct_pd_h *pd_p, uct_pd_component_t *pdc)
{
    uct_mm_pd_t *mm_pd;
    ucs_status_t status;

    mm_pd = ucs_malloc(sizeof(*mm_pd), "uct_mm_pd_t");
    if (mm_pd == NULL) {
        ucs_error("Failed to allocate memory for uct_mm_pd_t");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    mm_pd->config = ucs_malloc(pdc->pd_config_size, "mm_pd config");
    if (mm_pd->config == NULL) {
        ucs_error("Failed to allocate memory for mm_pd config");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_mm_pd;
    }

    status = ucs_config_parser_clone_opts(pd_config, mm_pd->config,
                                          pdc->pd_config_table);
    if (status != UCS_OK) {
        ucs_error("Failed to clone opts");
        goto err_free_mm_pd_config;
    }

    mm_pd->super.ops = &uct_mm_pd_ops;
    mm_pd->super.component = pdc;

    *pd_p = &mm_pd->super;
    return UCS_OK;

err_free_mm_pd_config:
    ucs_free(mm_pd->config);
err_free_mm_pd:
    ucs_free(mm_pd);
err:
    return status;
}
