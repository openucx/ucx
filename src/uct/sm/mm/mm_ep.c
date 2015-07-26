/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "mm_ep.h"


static UCS_CLASS_INIT_FUNC(uct_mm_ep_t, uct_iface_t *tl_iface,
                           const struct sockaddr *addr)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    uct_sockaddr_process_t *remote_iface_addr = (uct_sockaddr_process_t *)addr;
    ucs_status_t status;
    void *ptr;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    /* Connect to the remote address */
    /* Attach the address's memory */
    status = uct_mm_pd_mapper_ops(iface->super.pd)->attach(remote_iface_addr->id, &ptr);
    if (status != UCS_OK) {
        ucs_error("failed to connect to remote peer with mm. remote mm_id: %zu",
                   remote_iface_addr->id);
        return status;
    }

    self->fifo_ctl = ptr;
    self->fifo     = (void*) self->fifo_ctl + UCT_MM_FIFO_CTL_SIZE_ALIGNED;

    ucs_debug("mm: ep connected: %p, to remote_shmid: %zu", self, remote_iface_addr->id);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_mm_ep_t)
{
    uct_mm_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_mm_iface_t);
    ucs_status_t status;

    /* detach the remote proceess's shared memory segment (remote recv FIFO) */
    status = uct_mm_pd_mapper_ops(iface->super.pd)->release(self->fifo_ctl);
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

inline uint64_t uct_mm_ep_head_index_in_fifo(uct_mm_ep_t *ep)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_mm_iface_t);
    return (ep->fifo_ctl->head & iface->fifo_mask);
}

uct_mm_fifo_element_t* uct_mm_ep_get_remote_elem(uct_mm_ep_t *ep, uint64_t head)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_mm_iface_t);
    uct_mm_fifo_element_t* write_elem;  /* the fifo_element which the head points to */
    uint64_t elem_index;                /* the fifo elem's index in the fifo. */
                                        /* must be smaller than fifo size */
    uint64_t returned_val;

    elem_index = uct_mm_ep_head_index_in_fifo(ep);
    write_elem = (uct_mm_fifo_element_t*)(ep->fifo + elem_index * (iface->elem_size));

    /* try to get ownership on the head element */
    returned_val = ucs_atomic_cswap64(&ep->fifo_ctl->head, head, head+1);
    if (returned_val != head) {
        return NULL;
    }

    return write_elem;
}

ucs_status_t uct_mm_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);
    uct_mm_fifo_element_t *elem_to_write;
    uint64_t head;

    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(length + sizeof(header),
                     iface->elem_size - sizeof(uct_mm_fifo_element_t), "am_short");

    head = ep->fifo_ctl->head;
    /* check if there is room in the remote process's receive fifo to write */
    if ((head - ep->fifo_ctl->tail) >= iface->config.fifo_size) {
        return UCS_ERR_NO_RESOURCE;
    }

    elem_to_write = uct_mm_ep_get_remote_elem(ep, head);

    if (elem_to_write != NULL ) {
        /* write to the remote fifo */
        *(uint64_t*)(elem_to_write + 1) = header;
        memcpy((void*)(elem_to_write + 1) + sizeof(header), payload, length);
        elem_to_write->am_id  = id;
        elem_to_write->flags |= UCT_MM_FIFO_ELEM_FLAG_INLINE;
        elem_to_write->length = length + sizeof(header);

        /* memory barrier - make sure that the memory is flushed before setting the
         * 'writing is complete' flag which the reader checks */
        uct_mm_flush();

        /* change the owner bit to indicate that the writing is complete */
        if (head & iface->config.fifo_size) {
            elem_to_write->flags |= UCT_MM_FIFO_ELEM_FLAG_OWNER;
        } else {
            elem_to_write->flags &= ~UCT_MM_FIFO_ELEM_FLAG_OWNER;
        }
    } else {
        /* couldn't get an available fifo element */
        return UCS_ERR_NO_RESOURCE;
    }
    ucs_trace_data("MM: AM_SHORT am_id: %d buf=%p length=%u", id, payload, length);
    return UCS_OK;
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
