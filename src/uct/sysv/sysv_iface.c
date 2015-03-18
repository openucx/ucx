/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ucs/type/class.h"
#include "uct/tl/context.h"

#include "sysv_iface.h"
#include "sysv_ep.h"

unsigned sysv_iface_global_counter = 0;

static ucs_status_t uct_sysv_iface_flush(uct_iface_h tl_iface)
{
    return UCS_OK;
}

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_sysv_iface_t)(uct_iface_t*);

ucs_status_t uct_sysv_iface_get_address(uct_iface_h tl_iface, 
                                       uct_iface_addr_t *iface_addr)
{
    uct_sysv_iface_t *iface = ucs_derived_of(tl_iface, uct_sysv_iface_t);

    *(uct_sysv_iface_addr_t*)iface_addr = iface->addr;
    return UCS_OK;
}

#define UCT_SYSV_MAX_SHORT_LENGTH 2048 /* FIXME temp value for now */

ucs_status_t uct_sysv_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* FIXME all of these values */
    iface_attr->cap.put.max_short      = UCT_SYSV_MAX_SHORT_LENGTH;
    iface_attr->iface_addr_len         = sizeof(uct_sysv_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_sysv_ep_addr_t);
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT;

    iface_attr->completion_priv_len    = 0; /* TBD */
    return UCS_OK;
}

static ucs_status_t uct_sysv_mem_alloc(uct_pd_h pd, size_t *length_p, void **address_p,
                                       uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    ucs_status_t rc;
    uct_sysv_key_t *key_hndl = NULL;
    int shmid = 0;

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        return UCS_ERR_INVALID_PARAM;
    }

    key_hndl = ucs_malloc(sizeof(uct_sysv_key_t), "key_hndl");
    if (NULL == key_hndl) {
        ucs_error("Failed to allocate memory for key_hndl");
        return UCS_ERR_NO_MEMORY;
    }

    /* FIXME is this the right usage of ucs_sysv_alloc? */
    rc = ucs_sysv_alloc(length_p, address_p, 0, &shmid UCS_MEMTRACK_VAL);
    if (rc != UCS_OK) {
        ucs_error("Failed to attach %zu bytes", *length_p);
        return rc;
    }

    key_hndl->shmid = shmid;
    key_hndl->owner_ptr = (uintptr_t) *address_p;

    /* FIXME no use for pd input argument? */

    ucs_debug("Memory registration address_p %p, len %lu, keys [%d %"PRIx64"]",
              *address_p, *length_p, key_hndl->shmid, key_hndl->owner_ptr);
    *memh_p = key_hndl;
    ucs_memtrack_allocated(address_p, length_p UCS_MEMTRACK_VAL);
    return UCS_OK;
}

static ucs_status_t uct_sysv_mem_free(uct_pd_h pd, uct_mem_h memh)
{
    /* this releases the key allocated in uct_sysv_mem_alloc */

    uct_sysv_key_t *key_hndl = memh;
    ucs_sysv_free((void *)key_hndl->owner_ptr);  /* detach the shared segment */
    ucs_free(key_hndl);

    return UCS_OK;
}

#define UCT_SYSV_RKEY_MAGIC  0xabbadabaLL

static ucs_status_t uct_sysv_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    uct_sysv_pd_t *sysv_pd = ucs_derived_of(pd, uct_sysv_pd_t);

    ucs_snprintf_zero(pd_attr->name, UCT_MAX_NAME_LEN, "%s",
                      sysv_pd->iface->ctx->type_name);
    pd_attr->rkey_packed_size  = sizeof(uct_sysv_key_t);
    pd_attr->cap.flags         = UCT_PD_FLAG_ALLOC;
    pd_attr->cap.max_alloc     = ULONG_MAX;
    pd_attr->cap.max_reg       = 0;
    pd_attr->alloc_methods.count = 1;
    pd_attr->alloc_methods.methods[0] = UCT_ALLOC_METHOD_PD;
    return UCS_OK;
}

static ucs_status_t uct_sysv_rkey_pack(uct_pd_h pd, uct_mem_h memh,
                                      void *rkey_buffer)
{
    /* user is responsible to free rkey_buffer */
    uct_sysv_key_t *rkey = rkey_buffer;
    uct_sysv_key_t *key_hndl = memh;

    rkey->magic = UCT_SYSV_RKEY_MAGIC;
    rkey->shmid = key_hndl->shmid;
    rkey->owner_ptr = key_hndl->owner_ptr;

    rkey->client_ptr = 0; /* will be attached addr on the remote PE - obtained at unpack */
    ucs_debug("Packed [ %d %llx %"PRIx64" ]", rkey->shmid, rkey->magic, rkey->owner_ptr); 
    return UCS_OK;
}

static void uct_sysv_rkey_release(uct_context_h context, uct_rkey_t key)
{
    /* this releases the key allocated in unpack */
    
    uct_sysv_key_t *key_hndl = (uct_sysv_key_t *)key;

    ucs_sysv_free((void *)key_hndl->client_ptr);  /* detach the shared segment */
    ucs_free(key_hndl);
}

ucs_status_t uct_sysv_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                  uct_rkey_bundle_t *rkey_ob)
{
    /* user is responsible to free rkey_buffer */
    uct_sysv_key_t *rkey = rkey_buffer;
    long long magic = 0;
    uct_sysv_key_t *key_hndl = NULL;
    int shmid;

    ucs_debug("Unpacking[ %d %llx %"PRIx64" ]", rkey->shmid, rkey->magic, rkey->owner_ptr); 
    if (rkey->magic != UCT_SYSV_RKEY_MAGIC) {
        ucs_debug("Failed to identify key. Expected %llx but received %llx",
                  UCT_SYSV_RKEY_MAGIC, magic);
        return UCS_ERR_UNSUPPORTED;
    }

    key_hndl = ucs_malloc(sizeof(uct_sysv_key_t), "key_hndl");
    if (NULL == key_hndl) {
        ucs_error("Failed to allocate memory for key_hndl");
        return UCS_ERR_NO_MEMORY;
    }

    /* Attach segment */ 
    /* FIXME would like to extend ucs_sysv_alloc to do this? */
    shmid = rkey->shmid;
    rkey->client_ptr = (uintptr_t) shmat(shmid, NULL, 0);
    /* Check if attachment was successful */
    if ((void *)rkey->client_ptr == (void*)-1) {
        if (errno == ENOMEM) {
            return UCS_ERR_NO_MEMORY;
        } else if (RUNNING_ON_VALGRIND && (errno == EINVAL)) {
            return UCS_ERR_NO_MEMORY;
        } else {
            ucs_error("shmat(shmid=%d) returned unexpected error: %m", shmid);
            return UCS_ERR_SHMEM_SEGMENT;
        }
    }

    key_hndl->shmid      = rkey->shmid;
    key_hndl->magic      = rkey->magic;
    key_hndl->owner_ptr  = rkey->owner_ptr;
    key_hndl->client_ptr = rkey->client_ptr;

    rkey_ob->type = (void*)uct_sysv_rkey_release;
    rkey_ob->rkey = (uct_rkey_t)key_hndl;
    return UCS_OK;

}

uct_iface_ops_t uct_sysv_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_sysv_iface_t),
    .iface_get_address   = uct_sysv_iface_get_address,
    .iface_flush         = uct_sysv_iface_flush,
    .ep_get_address      = uct_sysv_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_sysv_ep_connect_to_ep,
    .iface_query         = uct_sysv_iface_query,
    .ep_put_short        = uct_sysv_ep_put_short,
    .ep_am_short         = uct_sysv_ep_am_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_sysv_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_sysv_ep_t),
};

uct_pd_ops_t uct_sysv_pd_ops = {
    .query        = uct_sysv_pd_query,
    .mem_alloc    = uct_sysv_mem_alloc,
    .mem_free     = uct_sysv_mem_free,
    .rkey_pack    = uct_sysv_rkey_pack,
};

static UCS_CLASS_INIT_FUNC(uct_sysv_iface_t, uct_context_h context,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_sysv_context_t *sysv_ctx = 
        ucs_component_get(context, sysv, uct_sysv_context_t);
    int addr;

    UCS_CLASS_CALL_SUPER_INIT(&uct_sysv_iface_ops);

    if(strcmp(dev_name, sysv_ctx->type_name) != 0) {
        ucs_error("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    /* FIXME initialize structure contents 
     * most of these copied from ugni tl iface code */
    self->pd.super.ops = &uct_sysv_pd_ops;
    self->pd.super.context = context;
    self->pd.iface = self;

    self->super.super.pd   = &self->pd.super;
    self->ctx              = sysv_ctx;
    self->config.max_put   = UCT_SYSV_MAX_SHORT_LENGTH;

    /* FIXME no use for config input argument? */

    addr = ucs_atomic_fadd32(&sysv_iface_global_counter, 1);

    self->addr.nic_addr = addr;
    self->activated = true;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sysv_iface_t)
{
    if (!self->activated) {
        /* We done with release */
        return;
    }

    /* TBD: Clean endpoints first (unbind and destroy) ?*/
    ucs_atomic_add32(&sysv_iface_global_counter, -1);

    /* tasks to tear down the domain */

    self->activated = false;
}

UCS_CLASS_DEFINE(uct_sysv_iface_t, uct_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_sysv_iface_t, uct_iface_t, uct_context_h,
                                 const char*, size_t, uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sysv_iface_t, uct_iface_t);

uct_tl_ops_t uct_sysv_tl_ops = {
    .query_resources     = uct_sysv_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_sysv_iface_t),
    .rkey_unpack         = uct_sysv_rkey_unpack,
};
