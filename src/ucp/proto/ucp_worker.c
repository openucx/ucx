/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_int.h"


static void ucp_worker_close_ifaces(ucp_worker_h worker)
{
    ucp_rsc_index_t rsc_index;

    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index] == NULL) {
            continue;
        }

        while (uct_iface_flush(worker->ifaces[rsc_index]) != UCS_OK) {
            uct_worker_progress(worker->uct);
        }
        uct_iface_close(worker->ifaces[rsc_index]);
    }
}

static ucs_status_t ucp_worker_add_iface(ucp_worker_h worker,
                                         ucp_rsc_index_t rsc_index)
{
    ucp_context_h context = worker->context;
    ucp_tl_resource_desc_t *resource = &context->tl_rscs[rsc_index];
    uct_iface_config_t *iface_config;
    ucs_status_t status;
    uct_iface_h iface;

    /* Read configuration */
    status = uct_iface_config_read(resource->tl_rsc.tl_name,
                                   UCP_CONFIG_ENV_PREFIX, NULL, &iface_config);
    if (status != UCS_OK) {
        goto out;
    }

    /* Open UCT interface */
    status = uct_iface_open(context->pds[resource->pd_index], worker->uct,
                            resource->tl_rsc.tl_name, resource->tl_rsc.dev_name,
                            sizeof(ucp_recv_desc_t), iface_config, &iface);
    uct_config_release(iface_config);

    if (status != UCS_OK) {
        goto out;
    }

    status = uct_iface_query(iface, &worker->iface_attrs[rsc_index]);
    if (status != UCS_OK) {
        goto out;
    }

    /* Set active message handlers for tag matching */
    status = ucp_tag_set_am_handlers(worker, iface);
    if (status != UCS_OK) {
        goto out_close_iface;
    }

    status = ucp_wireup_set_am_handlers(worker, iface);
    if (status != UCS_OK) {
        goto out_close_iface;
    }

    ucs_debug("created interface[%d] using "UCT_TL_RESOURCE_DESC_FMT" on worker %p",
              rsc_index, UCT_TL_RESOURCE_DESC_ARG(&resource->tl_rsc), worker);

    worker->ifaces[rsc_index] = iface;
    return UCS_OK;

out_close_iface:
    uct_iface_close(iface);
out:
    return status;
}

ucs_status_t ucp_worker_create(ucp_context_h context, ucs_thread_mode_t thread_mode,
                               ucp_worker_h *worker_p)
{
    ucp_rsc_index_t rsc_index;
    ucp_worker_h worker;
    ucs_status_t status;

    worker = ucs_calloc(1,
                        sizeof(*worker) + sizeof(worker->ifaces[0]) * context->num_tls,
                        "ucp worker");
    if (worker == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    worker->user_cb       = 0;
    worker->user_cb_arg   = 0;
    worker->context       = context;
    worker->uuid          = ucs_generate_uuid((uintptr_t)worker);

    worker->ep_hash = ucs_malloc(sizeof(*worker->ep_hash) * UCP_EP_HASH_SIZE,
                                 "ucp_ep_hash");
    if (worker->ep_hash == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    sglib_hashed_ucp_ep_t_init(worker->ep_hash);

    worker->iface_attrs = ucs_calloc(context->num_tls,
                                     sizeof(*worker->iface_attrs),
                                     "ucp iface_attr");
    if (worker->iface_attrs == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_ep_hash;
    }

    status = ucs_async_context_init(&worker->async, UCS_ASYNC_MODE_THREAD);
    if (status != UCS_OK) {
        goto err_free_attrs;
    }

    /* Create the underlying UCT worker */
    status = uct_worker_create(&worker->async, thread_mode, &worker->uct);
    if (status != UCS_OK) {
        goto err_destroy_async;
    }

    /* Open all resources as interfaces on this worker */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        status = ucp_worker_add_iface(worker, rsc_index);
        if (status != UCS_OK) {
            goto err_close_ifaces;
        }
    }

    *worker_p = worker;
    return UCS_OK;

err_close_ifaces:
    ucp_worker_close_ifaces(worker);
    uct_worker_destroy(worker->uct);
err_destroy_async:
    ucs_async_context_cleanup(&worker->async);
err_free_attrs:
    ucs_free(worker->iface_attrs);
err_free_ep_hash:
    ucs_free(worker->ep_hash);
err_free:
    ucs_free(worker);
err:
    return status;
}

static void ucp_worker_destroy_eps(ucp_worker_h worker)
{
    struct sglib_hashed_ucp_ep_t_iterator iter;
    ucp_ep_h ep;

    for (ep = sglib_hashed_ucp_ep_t_it_init(&iter, worker->ep_hash); ep != NULL;
         ep = sglib_hashed_ucp_ep_t_it_next(&iter))
    {
        ucp_ep_destroy(ep);
    }
}

void ucp_worker_destroy(ucp_worker_h worker)
{
    ucs_trace_func("worker=%p", worker);
    ucp_worker_destroy_eps(worker);
    ucp_worker_close_ifaces(worker);
    uct_worker_destroy(worker->uct);
    ucs_async_context_cleanup(&worker->async);
    ucs_free(worker->iface_attrs);
    ucs_free(worker->ep_hash);
    ucs_free(worker);
}

void ucp_worker_progress_register(ucp_worker_h worker,
                                  ucp_user_progress_func_t func, void *arg)
{
    worker->user_cb     = func;
    worker->user_cb_arg = arg;
}

void ucp_worker_progress(ucp_worker_h worker)
{
    uct_worker_progress(worker->uct);
    if (worker->user_cb != NULL) {
        /* TODO add to UCT callback chain */
        worker->user_cb(worker->user_cb_arg);
    }
    ucs_async_check_miss(&worker->async);
}

static ucs_status_t
ucp_worker_pack_resource_address(ucp_worker_h worker, ucp_rsc_index_t rsc_index,
                                 void **addr_buf_p, size_t *length_p)
{
    ucp_context_h context = worker->context;
    uct_iface_attr_t *iface_attr = &worker->iface_attrs[rsc_index];
    ucp_tl_resource_desc_t *resource = &context->tl_rscs[rsc_index];
    ucs_status_t status;
    void *buffer, *ptr;
    uint8_t tl_name_len;
    size_t length;
    int af;

    tl_name_len = strlen(resource->tl_rsc.tl_name);

    /* Calculate new address buffer size */
    length   = 1 +                           /* address length */
               iface_attr->iface_addr_len +  /* address */
               1 +                           /* tl name length */
               tl_name_len +                 /* tl name */
               1 +                           /* pd index */
               1;                            /* resource index */

    /* Enlarge address buffer */
    buffer = ucs_malloc(length, "ucp address");
    if (buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ptr = buffer;

    /* Copy iface address length */
    ucs_assert_always(iface_attr->iface_addr_len <= UINT8_MAX);
    ucs_assert_always(iface_attr->iface_addr_len > 0);
    *((uint8_t*)ptr++) = iface_attr->iface_addr_len;

    /* Copy iface address */
    status = uct_iface_get_address(worker->ifaces[rsc_index], ptr);
    if (status != UCS_OK) {
        goto err_free;
    }

    af   = ((struct sockaddr*)ptr)->sa_family;
    ptr += iface_attr->iface_addr_len;

    /* Transport name */
    *((uint8_t*)ptr++) = tl_name_len;
    memcpy(ptr, resource->tl_rsc.tl_name, tl_name_len);
    ptr += tl_name_len;

    *((ucp_rsc_index_t*)ptr++) = resource->pd_index;
    *((ucp_rsc_index_t*)ptr++) = rsc_index;

    ucs_trace("adding resource[%d]: "UCT_TL_RESOURCE_DESC_FMT" family %d address length %zu",
               rsc_index, UCT_TL_RESOURCE_DESC_ARG(&resource->tl_rsc), af, length);

    *addr_buf_p = buffer;
    *length_p   = length;
    return UCS_OK;

err_free:
    ucs_free(buffer);
err:
    return status;
}

ucs_status_t ucp_worker_get_address(ucp_worker_h worker, ucp_address_t **address_p,
                                    size_t *address_length_p)
{
    ucp_context_h context = worker->context;
    ucp_address_t *address;
    size_t address_length, rsc_addr_length;
    ucs_status_t status;
    ucp_rsc_index_t rsc_index;
    void *rsc_addr;

    address_length = sizeof(uint64_t);
    address        = ucs_malloc(address_length + 1, "ucp address");
    if (address == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    *(uint64_t*)address = worker->uuid;

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        status = ucp_worker_pack_resource_address(worker, rsc_index,
                                                  &rsc_addr, &rsc_addr_length);
        if (status != UCS_OK) {
            goto err_free;
        }

        /* Enlarge address buffer, leave room for NULL terminator */
        address_length += rsc_addr_length;
        address = ucs_realloc(address, address_length + 1, "ucp address");
        if (address == NULL) {
            status = UCS_ERR_NO_MEMORY;
            ucs_free(rsc_addr);
            goto err_free;
        }

        /* Add the address of the current resource */
        memcpy((void*)address + address_length - rsc_addr_length, rsc_addr, rsc_addr_length);
        ucs_free(rsc_addr);
    }

    if (address_length == 0) {
        ucs_error("No valid transport found");
        status = UCS_ERR_NO_DEVICE;
        goto err_free;
    }

    /* The final NULL terminator */
    *((uint8_t*)address + address_length) = 0;
    ++address_length;
    ucs_debug("worker uuid 0x%"PRIx64" address length: %zu", worker->uuid, address_length);

    *address_p        = address;
    *address_length_p = address_length;
    return UCS_OK;

err_free:
    ucs_free(address);
    return status;
}

void ucp_worker_release_address(ucp_worker_h worker, ucp_address_t *address)
{
    ucs_free(address);
}

SGLIB_DEFINE_LIST_FUNCTIONS(ucp_ep_t, ucp_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(ucp_ep_t, UCP_EP_HASH_SIZE, ucp_ep_hash);
