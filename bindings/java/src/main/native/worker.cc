/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "worker.h"

#include <new> // bad_alloc exception

worker::worker(context* ctx, uint32_t cap, ucp_worker_params_t params) :
               jucx_context(ctx),  ucp_worker(nullptr),
               event_cnt(0), head(nullptr), comp_queue(nullptr) {
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
    comp_queue = std::unique_ptr<completion_queue>(new completion_queue(cap));
}

worker::~worker() {
    free_requests();
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

void worker::put_in_event_queue(const command& cmd) {
    comp_queue->add_completion(cmd);
    ++event_cnt;
}

void worker::add_to_request_list(jucx_request* request) {
    if (head) {
        request->next = head;
    }
    head = request;
}

int worker::progress() {
    ucp_worker_progress(ucp_worker);

    int ret = event_cnt;
    event_cnt = 0;
    comp_queue->switch_primary_queue();

    free_requests();

    return ret;
}

void worker::free_requests() {
    jucx_request* current;

    while (head && head->done == 1) {
        current = head;
        head = head->next;
        jucx_handler::request_init(current);
        ucp_request_free(current);
    }
}
