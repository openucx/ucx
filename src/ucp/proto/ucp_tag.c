/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_int.h"

#include <ucs/arch/arch.h>
#include <inttypes.h>
#include <string.h>


static UCS_F_ALWAYS_INLINE int ucp_tag_is_match(ucp_tag_t tag, ucp_tag_t exp_tag,
                                                ucp_tag_t tag_mask)
{
    /* The bits in which expected and actual tag differ, should not fall
     * inside the mask.
     */
    return ((tag ^ exp_tag) & tag_mask) == 0;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_matched(void *buffer, size_t buffer_length, ucp_tag_t recv_tag,
                void *recv_data, size_t recv_length, ucp_tag_recv_completion_t *comp)
{
    ucs_debug("matched tag 0x%"PRIx64" ", (uint64_t)recv_tag);
    if (ucs_unlikely(recv_length > buffer_length)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    memcpy(buffer, recv_data, recv_length);
    comp->rcvd_len   = recv_length;
    comp->sender_tag = recv_tag;
    return UCS_OK;
}

static ucs_status_t ucp_tag_eager_am_handler(void *arg, void *data, size_t length,
                                             void *desc)
{
    ucp_context_h context = arg;
    ucp_recv_desc_t *rdesc = desc;
    ucp_recv_request_t *rreq;
    ucs_queue_iter_t iter;
    ucp_tag_t tag;

    ucs_assert(length >= sizeof(ucp_tag_t));
    tag = *(ucp_tag_t*)data;

    /* Search in expected queue */
    iter = ucs_queue_iter_begin(&context->tag.expected);
    while (!ucs_queue_iter_end(&context->tag.expected, iter)) {
        rreq = ucs_container_of(*iter, ucp_recv_request_t, queue);
        if (ucp_tag_is_match(tag, rreq->tag, rreq->tag_mask)) {
            ucs_queue_del_iter(&context->tag.expected, iter);
            rreq->status = ucp_tag_matched(rreq->buffer, rreq->length, tag,
                                           data + sizeof(ucp_tag_t),
                                           length - sizeof(ucp_tag_t),
                                           &rreq->comp);
            return UCS_OK;
        }
    }

    if (data != rdesc + 1) {
        memcpy(rdesc + 1, data, length);
    }

    rdesc->length = length;
    ucs_queue_push(&context->tag.unexpected, &rdesc->queue);
    return UCS_INPROGRESS;
}

ucs_status_t ucp_tag_send(ucp_ep_h ep, const void *buffer, size_t length,
                          ucp_tag_t tag)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_t status;

retry:
    if (ucs_likely(length < ep->config.max_short_tag)) {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uint64_t));
        status = uct_ep_am_short(ep->uct_ep, UCP_AM_ID_EAGER_ONLY, tag,
                                 buffer, length);
        if (status == UCS_ERR_NO_RESOURCE) {
            ucp_worker_progress(worker);
            goto retry;
        }
        return status;
    }

    ucs_fatal("unsupported");
}

ucs_status_t ucp_tag_recv(ucp_worker_h worker, void *buffer,
                          size_t length, ucp_tag_t tag, uint64_t tag_mask,
                          ucp_tag_recv_completion_t *comp)
{
    ucp_context_h context = worker->context;
    ucp_recv_request_t rreq;
    ucs_queue_iter_t iter;
    ucp_recv_desc_t *rdesc;
    ucp_tag_t unexp_tag;
    ucs_status_t status;

    /* First, search in unexpected list */
    iter = ucs_queue_iter_begin(&context->tag.unexpected);
    while (!ucs_queue_iter_end(&context->tag.unexpected, iter)) {
        rdesc = ucs_container_of(*iter, ucp_recv_desc_t, queue);
        unexp_tag = *(ucp_tag_t*)(rdesc + 1);
        if (ucp_tag_is_match(unexp_tag, tag, tag_mask)) {
            ucs_queue_del_iter(&context->tag.unexpected, iter);
            status = ucp_tag_matched(buffer, length, unexp_tag,
                                     (void*)(rdesc + 1) + sizeof(ucp_tag_t),
                                     rdesc->length - sizeof(ucp_tag_t),
                                     comp);
            uct_iface_release_am_desc(rdesc);
            goto out;
         }
         iter = ucs_queue_iter_next(iter);
    }

    /* If not found on unexpected, wait until it arrives */
    rreq.status   = UCS_INPROGRESS;
    rreq.buffer   = buffer;
    rreq.length   = length;
    rreq.tag      = tag;
    rreq.tag_mask = tag_mask;
    ucs_queue_push(&context->tag.expected, &rreq.queue);

    do {
        uct_worker_progress(worker->uct);
        /* coverity[loop_condition] */
    } while (rreq.status == UCS_INPROGRESS);

    *comp  = rreq.comp;
    status = rreq.status;

out:
    return status;
}

ucs_status_t ucp_tag_init(ucp_context_h context)
{
    if (!(context->config.features & UCP_FEATURE_TAG)) {
        return UCS_OK;
    }

    ucs_queue_head_init(&context->tag.expected);
    ucs_queue_head_init(&context->tag.unexpected);
    return UCS_OK;
}

void ucp_tag_cleanup(ucp_context_h context)
{
    if (!(context->config.features & UCP_FEATURE_TAG)) {
        return;
    }
}

ucs_status_t ucp_tag_set_am_handlers(ucp_worker_h worker, uct_iface_h iface)
{
    ucs_status_t status;

    if (!(worker->context->config.features & UCP_FEATURE_TAG)) {
        return UCS_OK;
    }

    status = uct_iface_set_am_handler(iface, UCP_AM_ID_EAGER_ONLY,
                                      ucp_tag_eager_am_handler, worker->context);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}
