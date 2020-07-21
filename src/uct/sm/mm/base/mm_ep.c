/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mm_ep.h"

#include <ucs/arch/atomic.h>


/* send modes */
typedef enum {
    UCT_MM_SEND_AM_BCOPY,
    UCT_MM_SEND_AM_SHORT,
} uct_mm_send_op_t;


/* Check if the resources on the remote peer are available for sending to it.
 * i.e. check if the remote receive FIFO has room in it.
 * return 1 if can send.
 * return 0 if can't send.
 */
#define UCT_MM_EP_IS_ABLE_TO_SEND(_head, _tail, _fifo_size) \
    ucs_likely(((_head) - (_tail)) < (_fifo_size))


static UCS_F_NOINLINE ucs_status_t
uct_mm_ep_attach_remote_seg(uct_mm_ep_t *ep, uct_mm_seg_id_t seg_id,
                            size_t length, void **address_p)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                           uct_mm_iface_t);
    uct_mm_remote_seg_t *remote_seg;
    ucs_status_t status;
    khiter_t khiter;
    int khret;

    khiter = kh_put(uct_mm_remote_seg, &ep->remote_segs, seg_id, &khret);
    if (khret == -1) {
        ucs_error("failed to add remote segment to mm ep hash");
        return UCS_ERR_NO_MEMORY;
    }

    /* we expect the key would either be never used (=1) or deleted (=2) */
    ucs_assert_always((khret == 1) || (khret == 2));

    remote_seg = &kh_val(&ep->remote_segs, khiter);

    status = uct_mm_iface_mapper_call(iface, mem_attach, seg_id, length,
                                      ep->remote_iface_addr, remote_seg);
    if (status != UCS_OK) {
        kh_del(uct_mm_remote_seg, &ep->remote_segs, khiter);
        return status;
    }

    *address_p = remote_seg->address;
    ucs_debug("mm_ep %p: attached remote segment id 0x%"PRIx64" at %p cookie %p",
              ep, seg_id, remote_seg->address, remote_seg->cookie);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_mm_ep_get_remote_seg(uct_mm_ep_t *ep, uct_mm_seg_id_t seg_id, size_t length,
                         void **address_p)
{
    khiter_t khiter;

    /* fast path - segment is already present */
    khiter = kh_get(uct_mm_remote_seg, &ep->remote_segs, seg_id);
    if (ucs_likely(khiter != kh_end(&ep->remote_segs))) {
        *address_p = kh_val(&ep->remote_segs, khiter).address;
        return UCS_OK;
    }

    /* slow path - attach new segment */
    return uct_mm_ep_attach_remote_seg(ep, seg_id, length, address_p);
}


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

static UCS_CLASS_INIT_FUNC(uct_mm_ep_t, const uct_ep_params_t *params)
{
    uct_mm_iface_t            *iface = ucs_derived_of(params->iface, uct_mm_iface_t);
    uct_mm_md_t               *md    = ucs_derived_of(iface->super.super.md, uct_mm_md_t);
    const uct_mm_iface_addr_t *addr  = (const void *)params->iface_addr;
    ucs_status_t status;
    void *fifo_ptr;

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    kh_init_inplace(uct_mm_remote_seg, &self->remote_segs);
    ucs_arbiter_group_init(&self->arb_group);

    /* save remote md address */
    if (md->iface_addr_len > 0) {
        self->remote_iface_addr = ucs_malloc(md->iface_addr_len, "mm_md_addr");
        if (self->remote_iface_addr == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        memcpy(self->remote_iface_addr, addr + 1, md->iface_addr_len);
    } else {
        self->remote_iface_addr = NULL;
    }

    /* Attach the remote FIFO, use the same method as bcopy descriptors */
    status = uct_mm_ep_get_remote_seg(self, addr->fifo_seg_id,
                                      UCT_MM_GET_FIFO_SIZE(iface), &fifo_ptr);
    if (status != UCS_OK) {
        ucs_error("mm ep failed to connect to remote FIFO id 0x%"PRIx64 ": %s",
                  addr->fifo_seg_id, ucs_status_string(status));
        goto err_free_md_addr;
    }

    /* Initialize remote FIFO control structure */
    uct_mm_iface_set_fifo_ptrs(fifo_ptr, &self->fifo_ctl, &self->fifo_elems);
    self->cached_tail     = self->fifo_ctl->tail;
    self->signal.addrlen  = self->fifo_ctl->signal_addrlen;
    self->signal.sockaddr = self->fifo_ctl->signal_sockaddr;

    ucs_debug("created mm ep %p, connected to remote FIFO id 0x%"PRIx64,
              self, addr->fifo_seg_id);

    return UCS_OK;

err_free_md_addr:
    ucs_free(self->remote_iface_addr);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_mm_ep_t)
{
    uct_mm_iface_t  *iface = ucs_derived_of(self->super.super.iface, uct_mm_iface_t);
    uct_mm_remote_seg_t remote_seg;

    uct_mm_ep_pending_purge(&self->super.super, NULL, NULL);

    kh_foreach_value(&self->remote_segs, remote_seg, {
        uct_mm_iface_mapper_call(iface, mem_detach, &remote_seg);
    })

    ucs_free(self->remote_iface_addr);
    kh_destroy_inplace(uct_mm_remote_seg, &self->remote_segs);
}

UCS_CLASS_DEFINE(uct_mm_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_mm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_mm_ep_t, uct_ep_t);


static inline ucs_status_t uct_mm_ep_get_remote_elem(uct_mm_ep_t *ep, uint64_t head,
                                                     uct_mm_fifo_element_t **elem)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_mm_iface_t);
    uint64_t elem_index;       /* the fifo elem's index in the fifo. */
                               /* must be smaller than fifo size */
    uint64_t returned_val;

    elem_index = ep->fifo_ctl->head & iface->fifo_mask;
    *elem      = UCT_MM_IFACE_GET_FIFO_ELEM(iface, ep->fifo_elems, elem_index);

    /* try to get ownership of the head element */
    returned_val = ucs_atomic_cswap64(ucs_unaligned_ptr(&ep->fifo_ctl->head), head, head+1);
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
uct_mm_ep_am_common_send(uct_mm_send_op_t send_op, uct_mm_ep_t *ep,
                         uct_mm_iface_t *iface, uint8_t am_id, size_t length,
                         uint64_t header, const void *payload,
                         uct_pack_callback_t pack_cb, void *arg,
                         unsigned flags)
{
    uct_mm_fifo_element_t *elem;
    ucs_status_t status;
    void *base_address;
    uint8_t elem_flags;
    uint64_t head;

    UCT_CHECK_AM_ID(am_id);

retry:
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
        ucs_assert(status == UCS_ERR_NO_RESOURCE);
        ucs_trace_poll("couldn't get an available FIFO element. retrying");
        goto retry;
    }

    switch (send_op) {
    case UCT_MM_SEND_AM_SHORT:
        /* write to the remote FIFO */
        uct_am_short_fill_data(elem + 1, header, payload, length);

        elem_flags   = UCT_MM_FIFO_ELEM_FLAG_INLINE;
        elem->length = length + sizeof(header);

        uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_SEND, am_id,
                           elem + 1, length + sizeof(header), "TX: AM_SHORT");
        UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(header) + length);
        break;
    case UCT_MM_SEND_AM_BCOPY:
        /* write to the remote descriptor */
        /* get the base_address: local ptr to remote memory chunk after attaching to it */
        status = uct_mm_ep_get_remote_seg(ep, elem->desc.seg_id,
                                          elem->desc.seg_size, &base_address);
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }

        length       = pack_cb(UCS_PTR_BYTE_OFFSET(base_address,
                                                   elem->desc.offset),
                               arg);
        elem_flags   = 0;
        elem->length = length;

        uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_SEND, am_id,
                           UCS_PTR_BYTE_OFFSET(base_address, elem->desc.offset),
                           length, "TX: AM_BCOPY");
        UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);
        break;
    }

    elem->am_id = am_id;

    /* memory barrier - make sure that the memory is flushed before setting the
     * 'writing is complete' flag which the reader checks */
    ucs_memory_cpu_store_fence();

    /* set the owner bit to indicate that the writing is complete.
     * the owner bit flips after every FIFO wraparound */
    if (head & iface->config.fifo_size) {
        elem_flags |= UCT_MM_FIFO_ELEM_FLAG_OWNER;
    }
    elem->flags = elem_flags;

    if (ucs_unlikely(flags & UCT_SEND_FLAG_SIGNALED)) {
        uct_mm_ep_signal_remote(ep);
    }

    switch (send_op) {
    case UCT_MM_SEND_AM_SHORT:
        return UCS_OK;
    case UCT_MM_SEND_AM_BCOPY:
        return length;
    default:
        return UCS_ERR_INVALID_PARAM;
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

    return (ucs_status_t)uct_mm_ep_am_common_send(UCT_MM_SEND_AM_SHORT, ep,
                                                  iface, id, length, header,
                                                  payload, NULL, NULL, 0);
}

ssize_t uct_mm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id, uct_pack_callback_t pack_cb,
                           void *arg, unsigned flags)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);

    return uct_mm_ep_am_common_send(UCT_MM_SEND_AM_BCOPY, ep, iface, id, 0, 0,
                                    NULL, pack_cb, arg, flags);
}

static inline int uct_mm_ep_has_tx_resources(uct_mm_ep_t *ep)
{
    uct_mm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_mm_iface_t);
    return UCT_MM_EP_IS_ABLE_TO_SEND(ep->fifo_ctl->head, ep->cached_tail,
                                     iface->config.fifo_size);
}

ucs_status_t uct_mm_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n,
                                   unsigned flags)
{
    uct_mm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_mm_iface_t);
    uct_mm_ep_t *ep = ucs_derived_of(tl_ep, uct_mm_ep_t);

    /* check if resources became available */
    if (uct_mm_ep_has_tx_resources(ep)) {
        ucs_assert(ucs_arbiter_group_is_empty(&ep->arb_group));
        return UCS_ERR_BUSY;
    }

    UCS_STATIC_ASSERT(sizeof(uct_pending_req_priv_arb_t) <=
                      UCT_PENDING_REQ_PRIV_LEN);
    uct_pending_req_arb_group_push(&ep->arb_group, n);
    /* add the ep's group to the arbiter */
    ucs_arbiter_group_schedule(&iface->arbiter, &ep->arb_group);
    UCT_TL_EP_STAT_PEND(&ep->super);

    return UCS_OK;
}

ucs_arbiter_cb_result_t uct_mm_ep_process_pending(ucs_arbiter_t *arbiter,
                                                  ucs_arbiter_group_t *group,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg)
{
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_mm_ep_t *ep        = ucs_container_of(group, uct_mm_ep_t, arb_group);
    unsigned *count        = (unsigned*)arg;
    ucs_status_t status;

    /* update the local tail with its actual value from the remote peer
     * making sure that the pending sends would use the real tail value */
    uct_mm_ep_update_cached_tail(ep);

    if (!uct_mm_ep_has_tx_resources(ep)) {
        return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }

    ucs_trace_data("progressing pending request %p", req);
    status = req->func(req);
    ucs_trace_data("status returned from progress pending: %s",
                   ucs_status_string(status));

    if (status == UCS_OK) {
        (*count)++;
        /* sent successfully. remove from the arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    } else if (status == UCS_INPROGRESS) {
        (*count)++;
        /* sent but not completed, keep in the arbiter */
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    } else {
        /* couldn't send. keep this request in the arbiter until the next time
         * this function is called */
        return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }
}

static ucs_arbiter_cb_result_t uct_mm_ep_abriter_purge_cb(ucs_arbiter_t *arbiter,
                                                          ucs_arbiter_group_t *group,
                                                          ucs_arbiter_elem_t *elem,
                                                          void *arg)
{
    uct_mm_ep_t *ep                 = ucs_container_of(group, uct_mm_ep_t,
                                                       arb_group);
    uct_pending_req_t *req          = ucs_container_of(elem, uct_pending_req_t,
                                                       priv);
    uct_purge_cb_args_t *cb_args    = arg;
    uct_pending_purge_callback_t cb = cb_args->cb;

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

    if (!uct_mm_ep_has_tx_resources(ep)) {
        if (!ucs_arbiter_group_is_empty(&ep->arb_group)) {
            return UCS_ERR_NO_RESOURCE;
        } else {
            uct_mm_ep_update_cached_tail(ep);
            if (!uct_mm_ep_has_tx_resources(ep)) {
                return UCS_ERR_NO_RESOURCE;
            }
        }
    }

    ucs_memory_cpu_store_fence();
    UCT_TL_EP_STAT_FLUSH(&ep->super);
    return UCS_OK;
}
