/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "request_util.h"
#include "worker.h"

void request_util::request_handler::request_init(void *request) {
    jucx_request* req       = (jucx_request*) request;

    req->cmd.init();
    req->request_worker     = nullptr;
    req->request_endpoint   = nullptr;
    req->done               = -1;
    req->next               = nullptr;
}

void request_util::request_handler::recv_completion_handler(void* request,
                                                            ucs_status_t status,
                                                            size_t length) {
    jucx_request* req = (jucx_request*) request;

    req->cmd.length = length;
    stream_request_handler(req, status);
}

void request_util::request_handler::send_completion_handler(void* request,
                                                            ucs_status_t status) {
    jucx_request* req = (jucx_request*) request;
    stream_request_handler(req, status);
}

void request_util::request_handler::stream_request_handler(request_t* request,
                                                           ucs_status_t status) {
    if (ucs_likely(status == UCS_OK)) {
        request->cmd.comp_status = CompletionStatus::JUCX_OK;
    }
    else {
        request->cmd.comp_status = translate_status(status);
    }
    if (request->request_worker) {
        request->request_worker->put_in_event_queue(request->cmd);
        request->done = 1;
    }
    else {
        request->done = 0;
    }
}

int request_util::check_stream_request(request_t* request,
                                       worker* request_worker,
                                       ucp_ep_h request_endpoint,
                                       uint64_t request_id,
                                       CommandType type,
                                       size_t length) {
    if (UCS_PTR_STATUS(request) == UCS_OK) {
        command cmd(request_id, length, type);
        request_worker->put_in_event_queue(cmd);
    }
    else if (ucp_request_check_status(request) == UCS_INPROGRESS) {
        request->request_endpoint = request_endpoint;
        request->cmd.set(request_id, type);

        if (request->done == 0) {
            request_worker->put_in_event_queue(request->cmd);
        }

        request->request_worker = request_worker;
        request_worker->add_to_request_list(request);
    }
    else {
        return 1;
    }

    return 0;
}

CompletionStatus request_util::translate_status(ucs_status_t ucs_status) {
    switch (ucs_status) {
    case UCS_ERR_CANCELED:
        return CompletionStatus::JUCX_ERR_CANCELED;

    default:
        return CompletionStatus::JUCX_ERR;
    }
}
