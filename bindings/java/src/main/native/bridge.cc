/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "bridge.h"
#include "request_util.h"

#include <ucp/api/ucp.h>
#include "ucs/time/time.h"

#include <cstring>
#include <iostream>


#define ERR_EXIT(_msg, _ret)  do {                    \
                                print_error(_msg);   \
                                return _ret;         \
                            } while(0)


static JavaVM *cached_jvm;
static jfieldID field_buff = NULL;
static jfieldID field_addr = NULL;

static ucp_context_h cached_ctx = NULL;


static void print_error(const char* error);

// Create context when JNI first loads
static ucs_status_t create_context();

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void* reserved) {
    cached_jvm = jvm;
    JNIEnv *env;

    if (jvm->GetEnv((void **) &env, JNI_VERSION_1_8)) {
        ERR_EXIT("JNI version 1.8 or higher required.", JNI_ERR);
    }

    ucs_status_t status = create_context();
    if (status != UCS_OK) {
        ERR_EXIT("Failed to create UCP context.", JNI_ERR);
    }

    return JNI_VERSION_1_8;
}

static void print_error(const char* error_msg) {
    fprintf(stderr, "[ERROR] JUCX - %s: %s\n", __FILE__, error_msg);
}

static ucs_status_t create_context() {
    if (cached_ctx) {
        return UCS_OK;
    }

    ucp_params_t ucp_params = { 0 };
    ucp_config_t *config;
    ucs_status_t status;
    ucp_context_h ucp_context;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        return status;
    }

    ucp_params.features     = UCP_FEATURE_TAG;
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES      |
                              UCP_PARAM_FIELD_REQUEST_INIT  |
                              UCP_PARAM_FIELD_REQUEST_SIZE;

    ucp_params.request_size = sizeof(jucx_request);
    ucp_params.request_init = request_util::request_handler::request_init;

    status = ucp_init(&ucp_params, config, &ucp_context);
    ucp_config_release(config);

    cached_ctx = ucp_context;

    return status;
}
