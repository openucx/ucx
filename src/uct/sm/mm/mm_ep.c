/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "mm_ep.h"

#include <ucs/arch/atomic.h>

SGLIB_DEFINE_LIST_FUNCTIONS(uct_mm_remote_seg_t, uct_mm_remote_seg_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_mm_remote_seg_t,
                                        UCT_MM_BASE_ADDRESS_HASH_SIZE,
                                        uct_mm_remote_seg_hash)


/* send a signal to remote interface using Unix-domain socket */
static void uct_mm_ep_signal_remote(uct_mm_ep_t *ep)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_mm_iface_t);
    char dummy = 0;
    int ret;

    for (;;) {
        ret = sendto(iface->signal_fd, &dummy, sizeof(dummy), 0,
                     (const struct sockaddr*)&ep->signal.sockaddr,
                     ep->signal.addrlen);
        if (ucs_unlikely(ret < 0)) {
            if (errno == EINTR) {
                /* Interrupted system call - retry */
                continue;
            } if ((errno == EAGAIN) || (errno == ECONNREFUSED)) {
                /* If we failed to signal because buffer is full - ignore the error
                 * since it means the remote side would get a signal anyway.
                 * If the remote side is not there - ignore the error as well.
                 */
                ucs_trace("failed to send wakeup signal: %m");
                return;
            } else {
                ucs_warn("failed to send wakeup signal: %m");
                return;
            }
        } else {
            ucs_assert(ret == sizeof(dummy));
            ucs_trace("sent wakeup from socket %d to %p", iface->signal_fd,
                      (const struct sockaddr*)&ep->signal.sockaddr);
            return;
        }
    }
}

static UCS_CLASS_INIT_FUNC(uct_mm_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_iface, uct_mm_iface_t);
    const uct_mm_iface_addr_t *addr = (const void*)iface_addr;
    ucs_status_t status;
    size_t size_to_attach;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    /* Connect to the remote address (remote FIFO) */
    /* Attach the address's memory */
    size_to_attach = UCT_MM_GET_FIFO_SIZE(iface);
    status =
        uct_mm_md_mapper_ops(iface->super.md)->attach(addr->id,
                                                      size_to_attach,
                                                      (void *)addr->vaddr,
                                                      &self->mapped_desc.address,
                                                      &self->mapped_desc.cookie,
                                                      iface->path);
    if (status != UCS_OK) {
        ucs_error("failed to connect to remote peer with mm. remote mm_id: %zu",
                   addr->id);
        return status;
    }

    self->mapped_desc.length     = size_to_attach;
    self->mapped_desc.mmid       = addr->id;

    /* point the ep->fifo_ctl to the remote fifo.
      * it's an aligned pointer to the beginning of the ctl struct in the remote FIFO */
    self->fifo_ctl        = uct_mm_set_fifo_ctl(self->mapped_desc.address);
    self->cached_tail     = self->fifo_ctl->tail;
    self->signal.addrlen  = self->fifo_ctl->signal_addrlen;
    self->signal.sockaddr = self->fifo_ctl->signal_sockaddr;

    /* set the ep->fifo ptr to point to the beginning of the fifo elements at
     * the remote peer */
    uct_mm_set_fifo_elems_ptr(self->mapped_desc.address, &self->fifo);

    /* Initiate the hash which will keep the base_adresses of remote memory
     * chunks that hold the descriptors for bcopy. */
    sglib_hashed_uct_mm_remote_seg_t_init(self->remote_segments_hash);

    ucs_arbiter_group_init(&self->arb_group);

    ucs_debug("mm: ep connected: %p, to remote_shmid: %zu", self, addr->id);

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
            status = uct_mm_md_mapper_ops(iface->super.md)->detach(remote_seg);
            if (status != UCS_OK) {
                ucs_warn("Unable to detach shared memory segment of descriptors: %s",
                         ucs_status_string(status));
            }
            ucs_free(remote_seg);
    }

    /* detach the remote proceess's shared memory segment (remote recv FIFO) */
    status = uct_mm_md_mapper_ops(iface->super.md)->detach(&self->mapped_desc);
    if (status != UCS_OK) {
        ucs_error("error detaching from remote FIFO");
    }

    uct_mm_ep_pending_purge(&self->super.super, NULL, NULL);
}

UCS_CLASS_DEFINE(uct_mm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_mm_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_mm_ep_t, uct_ep_t);

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

        status = uct_mm_md_mapper_ops(iface->super.md)->attach(elem->desc_mmid,
                                                               elem->desc_mpool_size,
                                                               elem->desc_chunk_base_addr,
                                                               &remote_seg->address,
                                                               &remote_seg->cookie,
                                                               iface->path);
        if (status != UCS_OK) {
            ucs_fatal("Failed to attach to remote mmid:%zu. %s ",
                      elem->desc_mmid, ucs_status_string(status));
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

static inline void uct_mm_ep_update_cached_tail(uct_mm_ep_t *ep)
{
    ucs_memory_cpu_load_fence();
    ep->cached_tail = ep->fifo_ctl->tail;
}

/* A common mm active message sending function.
 * The first parameter indicates the origin of the call.
 * is_short = 1 - perform AM short sending
 * is_short = 0 - perform AM bcopy sending
 */
static UCS_F_ALWAYS_INLINE ssize_t
uct_mm_ep_am_common_send(unsigned is_short, uct_mm_ep_t *ep, uct_mm_iface_t *iface,
                         uint8_t am_id, size_t length, uint64_t header,
                         const void *payload, uct_pack_callback_t pack_cb, void *arg,
                         unsigned flags)
{
    uct_mm_fifo_element_t *elem;
    ucs_status_t status;
    void *base_address;
    uint64_t head;

    UCT_CHECK_AM_ID(am_id);

    head = ep->fifo_ctl->head;
    /* check if there is room in the remote process's receive FIFO to write */
    if (!UCT_MM_EP_IS_ABLE_TO_SEND(head, ep->cached_tail, iface->config.fifo_size)) {
        if (!ucs_arbiter_group_is_empty(&ep->arb_group)) {
            /* pending isn't empty. don't send now to prevent out-of-order sending */
            UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
            return UCS_ERR_NO_RESOURCE;
        } else {
            /* pending is empty */
            /* update the local copy of the tail to its actual value on the remote peer */
            uct_mm_ep_update_cached_tail(ep);
            if (!UCT_MM_EP_IS_ABLE_TO_SEND(head, ep->cached_tail, iface->config.fifo_size)) {
                UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
                return UCS_ERR_NO_RESOURCE;
            }
        }
    }

    status = uct_mm_ep_get_remote_elem(ep, head, &elem);
    if (status != UCS_OK) {
        ucs_trace_poll("couldn't get an available FIFO element");
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return status;
    }

    if (is_short) {
        /* AM_SHORT */
        /* write to the remote FIFO */
        *(uint64_t*) (elem + 1) = header;
        memcpy((void*) (elem + 1) + sizeof(header), payload, length);

        elem->flags |= UCT_MM_FIFO_ELEM_FLAG_INLINE;
        elem->length = length + sizeof(header);

        uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, am_id,
                           elem + 1, length + sizeof(header), "TX: AM_SHORT");
        UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(header) + length);
    } else {
        /* AM_BCOPY */
        /* write to the remote descriptor */
        /* get the base_address: local ptr to remote memory chunk after attaching to it */
        base_address = uct_mm_ep_attach_remote_seg(ep, iface, elem);
        length = pack_cb(base_address + elem->desc_offset, arg);

        elem->flags &= ~UCT_MM_FIFO_ELEM_FLAG_INLINE;
        elem->length = length;

        uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, am_id,
                           base_address + elem->desc_offset, length, "TX: AM_BCOPY");

        UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);
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

    if (ucs_unlikely(flags & UCT_SEND_FLAG_SIGNALED)) {
        uct_mm_ep_signal_remote(ep);
    }

    if (is_short) {
        return UCS_OK;
    } else {
        return length;
    }
}

ucs_status_t uct_mm_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);

    UCT_CHECK_LENGTH(length + sizeof(header), 0,
                     iface->config.fifo_elem_size - sizeof(uct_mm_fifo_element_t),
                     "am_short");

    return uct_mm_ep_am_common_send(UCT_MM_AM_SHORT, ep, iface, id, length,
                                    header, payload, NULL, NULL, 0);
}

ssize_t uct_mm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id, uct_pack_callback_t pack_cb,
                           void *arg, unsigned flags)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);

    return uct_mm_ep_am_common_send(UCT_MM_AM_BCOPY, ep, iface, id, 0, 0, NULL,
                                    pack_cb, arg, flags);
}

static inline int uct_mm_ep_has_tx_resources(uct_mm_ep_t *ep)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_mm_iface_t);
    return UCT_MM_EP_IS_ABLE_TO_SEND(ep->fifo_ctl->head, ep->cached_tail,
                                     iface->config.fifo_size);
}

ucs_status_t uct_mm_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);

    /* check if resources became available */
    if (uct_mm_ep_has_tx_resources(ep)) {
        return UCS_ERR_BUSY;
    }

    UCS_STATIC_ASSERT(sizeof(ucs_arbiter_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);

    ucs_arbiter_elem_init((ucs_arbiter_elem_t *)n->priv);
    /* add the request to the ep's arbiter_group (pending queue) */
    ucs_arbiter_group_push_elem(&ep->arb_group, (ucs_arbiter_elem_t*) n->priv);
    /* add the ep's group to the arbiter */
    ucs_arbiter_group_schedule(&iface->arbiter, &ep->arb_group);

    return UCS_OK;
}

ucs_arbiter_cb_result_t uct_mm_ep_process_pending(ucs_arbiter_t *arbiter,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg)
{
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_status_t status;
    uct_mm_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_mm_ep_t, arb_group);

    /* update the local tail with its actual value from the remote peer
     * making sure that the pending sends would use the real tail value */
    ucs_memory_cpu_load_fence();
    ep->cached_tail = ep->fifo_ctl->tail;

    if (!uct_mm_ep_has_tx_resources(ep)) {
        return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }

    status = req->func(req);
    ucs_trace_data("progress pending request %p returned %s", req,
                   ucs_status_string(status));

    if (status == UCS_OK) {
        /* sent successfully. remove from the arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    } else if (status == UCS_INPROGRESS) {
        /* sent but not completed, keep in the arbiter */
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    } else {
        /* couldn't send. keep this request in the arbiter until the next time
         * this function is called */
        return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }
}

static ucs_arbiter_cb_result_t uct_mm_ep_abriter_purge_cb(ucs_arbiter_t *arbiter,
                                                          ucs_arbiter_elem_t *elem,
                                                          void *arg)
{
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_purge_cb_args_t *cb_args    = arg;
    uct_pending_purge_callback_t cb = cb_args->cb;
    uct_mm_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem),
                                       uct_mm_ep_t, arb_group);
    if (cb != NULL) {
        cb(req, cb_args->arg);
    } else {
        ucs_warn("ep=%p canceling user pending request %p", ep, req);
    }
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_mm_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                             void *arg)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);
    uct_purge_cb_args_t  args = {cb, arg};

    ucs_arbiter_group_purge(&iface->arbiter, &ep->arb_group,
                            uct_mm_ep_abriter_purge_cb, &args);
}

ucs_status_t uct_mm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                             uct_completion_t *comp)
{
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);

    uct_mm_ep_update_cached_tail(ep);

    if (!uct_mm_ep_has_tx_resources(ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    ucs_memory_cpu_store_fence();
    UCT_TL_EP_STAT_FLUSH(&ep->super);
    return UCS_OK;
}
