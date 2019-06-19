/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_ucx_jucx_ucp_UcpContext.h"

#include <ucp/api/ucp.h>


/**
 * Bridge method for creating ucp_context from java
 */
JNIEXPORT jlong JNICALL
Java_org_ucx_jucx_ucp_UcpContext_createContextNative(JNIEnv *env, jclass cls,
                                                     jobject jucx_ctx_params)
{
    ucp_params_t ucp_params = { 0 };
    ucp_context_h ucp_context;
    jfieldID field;

    jclass jucx_param_class = env->GetObjectClass(jucx_ctx_params);
    field = env->GetFieldID(jucx_param_class, "fieldMask", "J");
    ucp_params.field_mask = env->GetLongField(jucx_ctx_params, field);

    if (ucp_params.field_mask & UCP_PARAM_FIELD_FEATURES) {
        field = env->GetFieldID(jucx_param_class, "features", "J");
        ucp_params.features = env->GetLongField(jucx_ctx_params, field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_MT_WORKERS_SHARED) {
        field = env->GetFieldID(jucx_param_class, "mtWorkersShared", "Z");
        ucp_params.mt_workers_shared = env->GetBooleanField(jucx_ctx_params,
                                                            field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_ESTIMATED_NUM_EPS) {
        field = env->GetFieldID(jucx_param_class, "estimatedNumEps", "J");
        ucp_params.estimated_num_eps = env->GetLongField(jucx_ctx_params,
                                                         field);
    }

    if (ucp_params.field_mask & UCP_PARAM_FIELD_TAG_SENDER_MASK) {
        field = env->GetFieldID(jucx_param_class, "tagSenderMask", "J");
        ucp_params.estimated_num_eps = env->GetLongField(jucx_ctx_params,
                                                         field);
    }

    ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_INIT |
                             UCP_PARAM_FIELD_REQUEST_SIZE;
    ucp_params.request_size = sizeof(struct jucx_context);
    ucp_params.request_init = jucx_request_init;

    ucs_status_t status = ucp_init(&ucp_params, NULL, &ucp_context);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
    return (native_ptr)ucp_context;
}


JNIEXPORT void JNICALL
Java_org_ucx_jucx_ucp_UcpContext_cleanupContextNative(JNIEnv *env, jclass cls,
                                                      jlong ucp_context_ptr)
{
    ucp_cleanup((ucp_context_h)ucp_context_ptr);
}
