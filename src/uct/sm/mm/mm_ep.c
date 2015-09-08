/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "mm_ep.h"

SGLIB_DEFINE_LIST_FUNCTIONS(uct_mm_remote_seg_t, uct_mm_remote_seg_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_mm_remote_seg_t,
                                        UCT_MM_BASE_ADDRESS_HASH_SIZE,
                                        uct_mm_remote_seg_hash)

static UCS_CLASS_INIT_FUNC(uct_mm_ep_t, uct_iface_t *tl_iface,
                           const struct sockaddr *addr)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    uct_sockaddr_process_t *remote_iface_addr = (uct_sockaddr_process_t *)addr;
    ucs_status_t status;
    size_t size_to_attach;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    /* Connect to the remote address (remote FIFO) */
    /* Attach the address's memory */
    size_to_attach = UCT_MM_GET_FIFO_SIZE(iface);
    status =
        uct_mm_pd_mapper_ops(iface->super.pd)->attach(remote_iface_addr->id,
                                                      size_to_attach,
                                                      (void *)remote_iface_addr->vaddr,
                                                      &self->mapped_desc.address,
                                                      &self->mapped_desc.cookie);
    if (status != UCS_OK) {
        ucs_error("failed to connect to remote peer with mm. remote mm_id: %zu",
                   remote_iface_addr->id);
        return status;
    }

    self->mapped_desc.length = size_to_attach;
    self->mapped_desc.mmid   = remote_iface_addr->id;
    uct_mm_set_fifo_ptrs(self->mapped_desc.address, &self->fifo_ctl, &self->fifo);

    /* Initiate the hash which will keep the base_adresses of remote memory
     * chunks that hold the descriptors for bcopy. */
    sglib_hashed_uct_mm_remote_seg_t_init(self->remote_segments_hash);

    ucs_debug("mm: ep connected: %p, to remote_shmid: %zu", self, remote_iface_addr->id);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_mm_ep_t)
{
    uct_mm_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_mm_iface_t);
    ucs_status_t status;
    uct_mm_remote_seg_t *remote_seg;
    struct sglib_hashed_uct_mm_remote_seg_t_iterator iter;

    for (remote_seg = sglib_hashed_uct_mm_remote_seg_t_it_init(&iter, self->remote_segments_hash);
         remote_seg != NULL; remote_seg = sglib_hashed_uct_mm_remote_seg_t_it_next(&iter)) {
            sglib_hashed_uct_mm_remote_seg_t_delete(self->remote_segments_hash, remote_seg);
            /* detach the remote proceess's descriptors segment */
            status = uct_mm_pd_mapper_ops(iface->super.pd)->detach(remote_seg);
            if (status != UCS_OK) {
                ucs_warn("Unable to detach shared memory segment of descriptors: %m");
            }
            ucs_free(remote_seg);
    }

    /* detach the remote proceess's shared memory segment (remote recv FIFO) */
    status = uct_mm_pd_mapper_ops(iface->super.pd)->detach(&self->mapped_desc);
    if (status != UCS_OK) {
        ucs_error("error detaching from remote FIFO");
    }
}

UCS_CLASS_DEFINE(uct_mm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_mm_ep_t, uct_ep_t, uct_iface_t*,
                          const struct sockaddr *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_mm_ep_t, uct_ep_t);


#define uct_mm_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

ucs_status_t uct_mm_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey)
{
    if (ucs_likely(length != 0)) {
        memcpy((void *)(rkey + remote_addr), buffer, length);
        uct_mm_trace_data(remote_addr, rkey, "PUT_SHORT [buffer %p size %u]",
                          buffer, length);
    } else {
        ucs_trace_data("PUT_SHORT [zero-length]");
    }
    return UCS_OK;
}

ucs_status_t uct_mm_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, size_t length, 
                                 uint64_t remote_addr, uct_rkey_t rkey)
{
    if (ucs_likely(length != 0)) {
        pack_cb((void *)(rkey + remote_addr), arg, length);
        uct_mm_trace_data(remote_addr, rkey, "PUT_BCOPY [size %zu]", length);
    } else {
        ucs_trace_data("PUT_BCOPY [zero-length]");
    }
    return UCS_OK;
}

void *uct_mm_ep_attach_remote_seg(uct_mm_ep_t *ep, uct_mm_iface_t *iface, uct_mm_fifo_element_t *elem)
{
    uct_mm_remote_seg_t *remote_seg, search;
    ucs_status_t status;

    /* take the mmid of the chunk that the desc belongs to, (the desc that the fifo_elem
     * is 'assigned' to), and check if the ep has already attached to it.
     */
    search.mmid = elem->desc_mmid;
    remote_seg = sglib_hashed_uct_mm_remote_seg_t_find_member(ep->remote_segments_hash, &search);
    if (remote_seg == NULL) {
        /* not in the hash. attach to the memory the mmid refers to. the attach call
         * will return the base address of the mmid's chunk -
         * save this base address in a hash table (which maps mmid to base address). */
        remote_seg = ucs_malloc(sizeof(*remote_seg), "mm_desc");
        if (remote_seg == NULL) {
            ucs_fatal("Failed to allocated memory for a remote segment identifier. %m");
        }

        status = uct_mm_pd_mapper_ops(iface->super.pd)->attach(elem->desc_mmid,
                                      elem->desc_mpool_size,
                                      elem->desc_chunk_base_addr,
                                      &remote_seg->address, &remote_seg->cookie);
        if (status != UCS_OK) {
            ucs_fatal("Failed to attach to remote mmid:%zu (error=%m)", elem->desc_mmid);
        }

        remote_seg->mmid   = elem->desc_mmid;
        remote_seg->length = elem->desc_mpool_size;

        /* put the base address into the ep's hash table */
        sglib_hashed_uct_mm_remote_seg_t_add(ep->remote_segments_hash, remote_seg);
    }

    return remote_seg->address;

}

static inline ucs_status_t uct_mm_ep_get_remote_elem(uct_mm_ep_t *ep, uint64_t head,
		                                             uct_mm_fifo_element_t **elem)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_mm_iface_t);
    uint64_t elem_index;       /* the fifo elem's index in the fifo. */
                               /* must be smaller than fifo size */
    uint64_t returned_val;

    elem_index = ep->fifo_ctl->head & iface->fifo_mask;
    *elem = UCT_MM_IFACE_GET_FIFO_ELEM(iface, ep->fifo, elem_index);

    /* try to get ownership of the head element */
    returned_val = ucs_atomic_cswap64(&ep->fifo_ctl->head, head, head+1);
    if (returned_val != head) {
        return UCS_ERR_NO_RESOURCE;
    }

    return UCS_OK;
}

/* A common mm active message sending function.
 * The first parameter indicates the origin of the call.
 * 1 - perform am short sending
 * 0 - perform am bcopy sending
 */
static UCS_F_ALWAYS_INLINE ucs_status_t uct_mm_ep_am_common_send(unsigned is_short,
                                        uct_mm_ep_t *ep, uct_mm_iface_t *iface,
                                        uint8_t am_id, unsigned length,
                                        uint64_t header, const void *payload,
                                        uct_pack_callback_t pack_cb, void *arg)
{
    uct_mm_fifo_element_t *elem;
    ucs_status_t status;
    uint64_t head;
    void *base_address;

    UCT_CHECK_AM_ID(am_id);

    head = ep->fifo_ctl->head;
    /* check if there is room in the remote process's receive fifo to write */
    if (ucs_unlikely((head - ep->fifo_ctl->tail) >= iface->config.fifo_size)) {
        return UCS_ERR_NO_RESOURCE;
    }

    status = uct_mm_ep_get_remote_elem(ep, head, &elem);

    if (status == UCS_OK) {
        if (is_short == 1) {
            /* AM_SHORT */
            /* write to the remote FIFO */
            *(uint64_t*) (elem + 1) = header;
            memcpy((void*) (elem + 1) + sizeof(header), payload, length);

            elem->flags |= UCT_MM_FIFO_ELEM_FLAG_INLINE;
            elem->length = length + sizeof(header);
        } else {
            /* AM_BCOPY */
            /* write to the remote descriptor */
            /* get the base_address: local ptr to remote memory chunk after attaching to it */
            base_address = uct_mm_ep_attach_remote_seg(ep, iface, elem);
            pack_cb(base_address + elem->desc_offset, arg, length);

            elem->flags &= ~UCT_MM_FIFO_ELEM_FLAG_INLINE;
            elem->length = length;

        }
        elem->am_id = am_id;

        /* memory barrier - make sure that the memory is flushed before setting the
         * 'writing is complete' flag which the reader checks */
        ucs_memory_cpu_store_fence();

        /* change the owner bit to indicate that the writing is complete.
         * the owner bit flips after every FIFO wraparound */
        if (head & iface->config.fifo_size) {
            elem->flags |= UCT_MM_FIFO_ELEM_FLAG_OWNER;
        } else {
            elem->flags &= ~UCT_MM_FIFO_ELEM_FLAG_OWNER;
        }
    }

    return status;
}

ucs_status_t uct_mm_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);
    ucs_status_t status;

    UCT_CHECK_LENGTH(length + sizeof(header),
                     iface->elem_size - sizeof(uct_mm_fifo_element_t), "am_short");

    status = uct_mm_ep_am_common_send(1, ep, iface, id, length, header, payload,
                                (void*)ucs_empty_function_return_success, (void*)0x0);
    if (status == UCS_OK) {
        ucs_trace_data("MM: AM_SHORT [%p] am_id: %d buf=%p length=%u",
                        iface, id, payload, length);
    } else {
        ucs_debug("Couldn't get an available FIFO element in am_short");
    }

    return status;
}

ucs_status_t uct_mm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg,
                                size_t length)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);
    ucs_status_t status;

    UCT_CHECK_LENGTH(length, iface->config.seg_size, "am_bcopy");

    status = uct_mm_ep_am_common_send(0, ep, iface, id, length, 0, (void*)0x0, pack_cb, arg);
    if (status == UCS_OK) {
        ucs_trace_data("MM: AM_BCOPY [%p] am_id: %d buf=%p length=%u",
                        iface, id, arg, (int)length);
    } else {
        ucs_debug("Couldn't get an available FIFO element in am_bcopy");
    }

    return status;
}

ucs_status_t uct_mm_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    ucs_atomic_add64(ptr, add);
    uct_mm_trace_data(remote_addr, rkey, "ATOMIC_ADD64 [add %"PRIu64"]", add);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_fadd64(ptr, add);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_FADD64 [add %"PRIu64" result %"PRIu64"]",
                      add, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_swap64(ptr, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_SWAP64 [swap %"PRIu64" result %"PRIu64"]",
                      swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp)
{
    uint64_t *ptr = (uint64_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap64(ptr, compare, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_CSWAP64 [compare %"PRIu64" swap %"PRIu64" result %"PRIu64"]",
                      compare, swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    ucs_atomic_add32(ptr, add);
    uct_mm_trace_data(remote_addr, rkey, "ATOMIC_ADD32 [add %"PRIu32"]", add);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_fadd32(ptr, add);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_FADD32 [add %"PRIu32" result %"PRIu32"]",
                      add, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_swap32(ptr, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_SWAP32 [swap %"PRIu32" result %"PRIu32"]",
                      swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp)
{
    uint32_t *ptr = (uint32_t *)(rkey + remote_addr);
    *result = ucs_atomic_cswap32(ptr, compare, swap);
    uct_mm_trace_data(remote_addr, rkey,
                      "ATOMIC_CSWAP32 [compare %"PRIu32" swap %"PRIu32" result %"PRIu32"]",
                      compare, swap, *result);
    return UCS_OK;
}

ucs_status_t uct_mm_ep_get_bcopy(uct_ep_h tl_ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp)
{
    if (ucs_likely(0 != length)) {
        unpack_cb(arg, (void *)(rkey + remote_addr), length);
        uct_mm_trace_data(remote_addr, rkey, "GET_BCOPY [length %zu]", length);
    } else {
        ucs_trace_data("GET_BCOPY [zero-length]");
    }
    return UCS_OK;
}
