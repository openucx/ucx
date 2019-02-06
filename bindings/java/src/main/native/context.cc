/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "helper.h"
#include "org_ucx_jucx_Bridge.h"

#include <ucp/api/ucp.h>
/**
 * Bridge methods for creating ucp_context from java
 */
JNIEXPORT jlong JNICALL
Java_org_ucx_jucx_Bridge_createContextNative(JNIEnv *env, jclass cls, jobject jucx_context_params)
{
    ucp_params_t ucp_params = { 0 };
    ucp_config_t *config;
    ucs_status_t status;
    ucp_context_h ucp_context;
    jfieldID field;
    jclass jucx_param_class = env->GetObjectClass(jucx_context_params);

    status = ucp_config_read(nullptr, nullptr, &config);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    field = env->GetFieldID(jucx_param_class, "fieldMask", "J");
    ucp_params.field_mask = env->GetLongField(jucx_context_params, field);

    if (ucp_params.field_mask & UCP_PARAM_FIELD_FEATURES) {
        field = env->GetFieldID(jucx_param_class, "features", "J");
        ucp_params.features = env->GetLongField(jucx_context_params, field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_MT_WORKERS_SHARED) {
        field = env->GetFieldID(jucx_param_class, "mtWorkersShared", "Z");
        ucp_params.mt_workers_shared = env->GetBooleanField(jucx_context_params, field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_ESTIMATED_NUM_EPS) {
        field = env->GetFieldID(jucx_param_class, "estimatedNumEps", "J");
        ucp_params.estimated_num_eps = env->GetLongField(jucx_context_params, field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_TAG_SENDER_MASK) {
        field = env->GetFieldID(jucx_param_class, "estimatedNumEps", "J");
        ucp_params.estimated_num_eps = env->GetLongField(jucx_context_params, field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_REQUEST_SIZE) {
        field = env->GetFieldID(jucx_param_class, "requestSize", "J");
        ucp_params.request_size = env->GetLongField(jucx_context_params, field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_REQUEST_INIT) {
        callback_wrapper init_wrapper;
        field = env->GetFieldID(jucx_param_class, "requestInit", "Lorg/ucx/jucx/Callback;");
        jobject jcallback = env->GetObjectField(jucx_param_class, field);

        jclass callback_class = env->FindClass("org/ucx/jucx/Callback");
        jmethodID jcallback_function = env->GetMethodID(callback_class, "onComplete", "(Ljava/nio/ByteBuffer;)V");

        init_wrapper.java_obj = jcallback;
        init_wrapper.java_callback = jcallback_function;
        init_wrapper.request_size = ucp_params.request_size;
        init_wrapper.env = env;
        //TODO: fix this - use callback that would call java callback
        ucp_params.request_init = init_wrapper.callback;
    }

     if (ucp_params.field_mask & UCP_PARAM_FIELD_REQUEST_CLEANUP) {
        callback_wrapper cleanup_wrapper;
        field = env->GetFieldID(jucx_param_class, "requestCleanup", "Lorg/ucx/jucx/Callback;");
        jobject jcallback = env->GetObjectField(jucx_param_class, field);

        jclass callback_class = env->FindClass("org/ucx/jucx/Callback");
        jmethodID jcallback_function = env->GetMethodID(callback_class, "onComplete", "(Ljava/nio/ByteBuffer;)V");

        cleanup_wrapper.java_obj = jcallback;
        cleanup_wrapper.java_callback = jcallback_function;
        cleanup_wrapper.request_size = ucp_params.request_size;
        cleanup_wrapper.env = env;
        //TODO: fix this - use callback that would call java callback
        ucp_params.request_cleanup = cleanup_wrapper.callback;
    }

    status = ucp_init(&ucp_params, config, &ucp_context);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    ucp_config_release(config);
    return (native_ptr) ucp_context;
}


JNIEXPORT void JNICALL
Java_org_ucx_jucx_Bridge_cleanupContextNative(JNIEnv *env, jclass cls, jlong ucp_context_ptr)
{
    ucp_cleanup((ucp_context_h)ucp_context_ptr);
}