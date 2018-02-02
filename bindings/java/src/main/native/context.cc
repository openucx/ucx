/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "context.h"
#include "request_util.h"

ucs_status_t context::ref_context() {
    ucs_status_t status = UCS_OK;
    {   // Lock before checking context and updating reference counter
        std::lock_guard<std::mutex> lk(ref_lock);
        if (ucp_context == nullptr) {
            status = create_context();
        }

        if (status == UCS_OK) {
            ++ref_count;
        }
    }   // Unlock

    return status;
}

void context::deref_context() {
    std::lock_guard<std::mutex> lk(ref_lock);
    if (--ref_count == 0) { // All workers released
        release_context();
    }
}

context::~context() {
    if (ucp_context) {
        ucp_cleanup(ucp_context);
    }
}

ucp_context_h context::get_ucp_context() {
    return ucp_context;
}

ucs_status_t context::create_context() {
    ucp_params_t ucp_params = { 0 };
    ucp_config_t *config;
    ucs_status_t status;

    status = ucp_config_read(nullptr, nullptr, &config);
    if (status != UCS_OK) {
        return status;
    }

    uint64_t features   =   UCP_FEATURE_TAG;
    uint64_t field_mask =   UCP_PARAM_FIELD_FEATURES        |
                            UCP_PARAM_FIELD_REQUEST_INIT    |
                            UCP_PARAM_FIELD_REQUEST_SIZE;

    ucp_params.features     = features;
    ucp_params.field_mask   = field_mask;
    ucp_params.request_size = sizeof(jucx_request);
    ucp_params.request_init = request_util::request_handler::request_init;

    status = ucp_init(&ucp_params, config, &ucp_context);
    ucp_config_release(config);

    return status;
}

void context::release_context() {
    if (ucp_context) {
        ucp_cleanup(ucp_context);
    }
    ucp_context = nullptr;
}
