/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "sysv_iface.h"

#include <uct/tl/context.h>


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
#define UCT_SYSV_MAX_BCOPY_LENGTH 40960 /* FIXME temp value for now */
#define UCT_SYSV_MAX_ZCOPY_LENGTH 81920 /* FIXME temp value for now */

ucs_status_t uct_sysv_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_sysv_iface_t *iface = ucs_derived_of(tl_iface, uct_sysv_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* FIXME all of these values */
    iface_attr->cap.put.max_short      = iface->config.max_put;
    iface_attr->cap.put.max_bcopy      = iface->config.max_bcopy;
    iface_attr->cap.put.max_zcopy      = iface->config.max_zcopy;
    iface_attr->cap.get.max_bcopy      = iface->config.max_bcopy;
    iface_attr->cap.get.max_zcopy      = iface->config.max_zcopy;
    iface_attr->iface_addr_len         = sizeof(uct_sysv_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_sysv_ep_addr_t);
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT       |
                                         UCT_IFACE_FLAG_PUT_BCOPY       |
                                         UCT_IFACE_FLAG_ATOMIC_ADD32    |
                                         UCT_IFACE_FLAG_ATOMIC_ADD64    |
                                         UCT_IFACE_FLAG_ATOMIC_FADD64   |
                                         UCT_IFACE_FLAG_ATOMIC_FADD32   |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP64   |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP32   |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP64  |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP32  |
                                         UCT_IFACE_FLAG_PUT_ZCOPY       |
                                         UCT_IFACE_FLAG_GET_BCOPY       |
                                         UCT_IFACE_FLAG_GET_ZCOPY;

    iface_attr->completion_priv_len    = 0; /* TBD */
    return UCS_OK;
}

static ucs_status_t uct_sysv_mem_alloc(uct_pd_h pd, size_t *length_p,
                                       void **address_p,
                                       uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    ucs_status_t rc;
    uct_sysv_lkey_t *key_hndl = NULL;
    int shmid = 0;

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        return UCS_ERR_INVALID_PARAM;
    }

    key_hndl = ucs_malloc(sizeof(*key_hndl), "key_hndl");
    if (NULL == key_hndl) {
        ucs_error("Failed to allocate memory for key_hndl");
        return UCS_ERR_NO_MEMORY;
    }

    rc = ucs_sysv_alloc(length_p, address_p, 0, &shmid UCS_MEMTRACK_VAL);
    if (rc != UCS_OK) {
        ucs_error("Failed to attach %zu bytes", *length_p);
        ucs_free(key_hndl);
        return rc;
    }

    key_hndl->shmid = shmid;
    key_hndl->owner_ptr = *address_p;

    ucs_debug("Memory registration address_p %p, len %lu, keys [%d %p]",
              *address_p, *length_p, key_hndl->shmid, key_hndl->owner_ptr);
    *memh_p = key_hndl;
    ucs_memtrack_allocated(address_p, length_p UCS_MEMTRACK_VAL);
    return UCS_OK;
}

static ucs_status_t uct_sysv_mem_free(uct_pd_h pd, uct_mem_h memh)
{
    /* this releases the key allocated in uct_sysv_mem_alloc */

    uct_sysv_lkey_t *key_hndl = memh;
    ucs_sysv_free(key_hndl->owner_ptr);  /* detach shared segment */
    ucs_free(key_hndl);

    return UCS_OK;
}

#define UCT_SYSV_RKEY_MAGIC  0xabbadabaLL

static ucs_status_t uct_sysv_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    ucs_snprintf_zero(pd_attr->name, sizeof(pd_attr->name), "%s", UCT_SYSV_TL_NAME);
    pd_attr->rkey_packed_size  = sizeof(uct_sysv_rkey_t);
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
    uct_sysv_rkey_t *rkey = rkey_buffer;
    uct_sysv_lkey_t *key_hndl = memh;

    rkey->magic = UCT_SYSV_RKEY_MAGIC;
    rkey->shmid = key_hndl->shmid;
    rkey->owner_ptr = (uintptr_t)key_hndl->owner_ptr;

    ucs_debug("Packed [ %d %llx %"PRIxPTR" ]",
              rkey->shmid, rkey->magic, rkey->owner_ptr);
    return UCS_OK;
}

static void uct_sysv_rkey_release(uct_pd_h pd, const uct_rkey_bundle_t *rkey_ob)
{
    /* detach shared segment */
    shmdt((void *)((intptr_t)rkey_ob->type + rkey_ob->rkey));
}

ucs_status_t uct_sysv_rkey_unpack(uct_pd_h pd, const void *rkey_buffer,
                                  uct_rkey_bundle_t *rkey_ob)
{
    /* user is responsible to free rkey_buffer */
    const uct_sysv_rkey_t *rkey = rkey_buffer;
    long long magic = 0;
    int shmid;
    void *client_ptr;

    ucs_debug("Unpacking[ %d %llx %"PRIxPTR" ]",
              rkey->shmid, rkey->magic, rkey->owner_ptr);
    if (rkey->magic != UCT_SYSV_RKEY_MAGIC) {
        ucs_debug("Failed to identify key. Expected %llx but received %llx",
                  UCT_SYSV_RKEY_MAGIC, magic);
        return UCS_ERR_UNSUPPORTED;
    }

    /* cache the local key on the rkey_ob */
    rkey_ob->type = (void *)rkey->owner_ptr;

    /* Attach segment */ 
    /* FIXME would like to extend ucs_sysv_alloc to do this? */
    shmid = rkey->shmid;
    client_ptr = shmat(shmid, NULL, 0);
    /* Check if attachment was successful */
    if (client_ptr == (void*)-1) {
        if (errno == ENOMEM) {
            return UCS_ERR_NO_MEMORY;
        } else if (RUNNING_ON_VALGRIND && (errno == EINVAL)) {
            return UCS_ERR_NO_MEMORY;
        } else {
            ucs_error("shmat(shmid=%d) returned unexpected error: %m", shmid);
            return UCS_ERR_SHMEM_SEGMENT;
        }
    }

    /* store the offset of the addresses */
    rkey_ob->rkey = (uintptr_t)client_ptr - rkey->owner_ptr;

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
    .ep_put_bcopy        = uct_sysv_ep_put_bcopy,
    .ep_put_zcopy        = uct_sysv_ep_put_zcopy,
    .ep_get_bcopy        = uct_sysv_ep_get_bcopy,
    .ep_get_zcopy        = uct_sysv_ep_get_zcopy,
    .ep_am_short         = uct_sysv_ep_am_short,
    .ep_atomic_add64     = uct_sysv_ep_atomic_add64,
    .ep_atomic_fadd64    = uct_sysv_ep_atomic_fadd64,
    .ep_atomic_cswap64   = uct_sysv_ep_atomic_cswap64,
    .ep_atomic_swap64    = uct_sysv_ep_atomic_swap64,
    .ep_atomic_add32     = uct_sysv_ep_atomic_add32,
    .ep_atomic_fadd32    = uct_sysv_ep_atomic_fadd32,
    .ep_atomic_cswap32   = uct_sysv_ep_atomic_cswap32,
    .ep_atomic_swap32    = uct_sysv_ep_atomic_swap32,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_sysv_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_sysv_ep_t),
};

static uct_pd_ops_t uct_sysv_pd_ops = {
    .query        = uct_sysv_pd_query,
    .mem_alloc    = uct_sysv_mem_alloc,
    .mem_free     = uct_sysv_mem_free,
    .rkey_pack    = uct_sysv_rkey_pack,
    .rkey_unpack  = uct_sysv_rkey_unpack,
    .rkey_release = uct_sysv_rkey_release
};

static uct_pd_t uct_sysv_pd = {
    .ops = &uct_sysv_pd_ops
};

static UCS_CLASS_INIT_FUNC(uct_sysv_iface_t, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    int addr;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_sysv_iface_ops, worker,
                              &uct_sysv_pd, tl_config UCS_STATS_ARG(NULL));

    if(strcmp(dev_name, UCT_SYSV_TL_NAME) != 0) {
        ucs_error("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->config.max_put     = UCT_SYSV_MAX_SHORT_LENGTH;
    self->config.max_bcopy   = UCT_SYSV_MAX_BCOPY_LENGTH;
    self->config.max_zcopy   = UCT_SYSV_MAX_ZCOPY_LENGTH;

    addr = ucs_generate_uuid((intptr_t)self);

    self->addr.nic_addr = addr;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sysv_iface_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_sysv_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_sysv_iface_t, uct_iface_t, uct_worker_h,
                                 const char*, size_t, const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sysv_iface_t, uct_iface_t);

uct_tl_ops_t uct_sysv_tl_ops = {
    .query_resources     = uct_sysv_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_sysv_iface_t),
};
