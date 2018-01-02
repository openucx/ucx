/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "worker.h"

#include <new> // bad_alloc exception

worker::worker(context* ctx, uint32_t cap, ucp_worker_params_t params) :
               jucx_context(ctx),  ucp_worker(nullptr),
               queue_size(cap),    event_queue(nullptr) {
    ucs_status_t status = jucx_context->ref_context();
    if (status != UCS_OK) {
        throw std::bad_alloc{};
    }

    status = ucp_worker_create(jucx_context->get_ucp_context(),
                               &params, &ucp_worker);
    if (status != UCS_OK) {
        jucx_context->deref_context();
        throw std::bad_alloc{};
    }
    event_queue = new char[queue_size];
}

worker::~worker() {
    delete[] event_queue;
    ucp_worker_destroy(ucp_worker);
    jucx_context->deref_context();
}

ucs_status_t worker::extract_worker_address(ucp_address_t** worker_address,
                                            size_t& address_length) {
    return ucp_worker_get_address(ucp_worker, worker_address, &address_length);
}

void worker::release_worker_address(ucp_address_t* worker_address) {
    ucp_worker_release_address(ucp_worker, worker_address);
}
