/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_ucx_jucx_ucp_UcpContext.h"
extern "C" {
#include <ucp/core/ucp_mm.h>
}

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

JNIEXPORT jobject JNICALL
Java_org_ucx_jucx_ucp_UcpContext_registerMemoryNative(JNIEnv *env, jobject ctx,
                                                      jlong ucp_context_ptr,
                                                      jobject maped_buf)
{
    ucp_mem_map_params_t params;
    ucp_mem_h memh;
    jfieldID field;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
    params.address    = env->GetDirectBufferAddress(maped_buf);
    params.length     = env->GetDirectBufferCapacity(maped_buf);

    ucs_status_t status =  ucp_mem_map((ucp_context_h)ucp_context_ptr, &params, &memh);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    // Construct UcpMemory class
    jclass jucx_mem_cls = env->FindClass("org/ucx/jucx/ucp/UcpMemory");
    jmethodID constructor = env->GetMethodID(jucx_mem_cls, "<init>", "(J)V");
    jobject jucx_mem = env->NewObject(jucx_mem_cls, constructor, (native_ptr)memh);

    // Set UcpContext pointer
    field = env->GetFieldID(jucx_mem_cls, "context", "Lorg/ucx/jucx/ucp/UcpContext;");
    env->SetObjectField(jucx_mem, field, ctx);

    // Set data buffer
    jobject data_buf = env->NewDirectByteBuffer(memh->address, memh->length);
    field = env->GetFieldID(jucx_mem_cls, "data", "Ljava/nio/ByteBuffer;");
    env->SetObjectField(jucx_mem, field, data_buf);

    // Set address
    field =  env->GetFieldID(jucx_mem_cls, "address", "J");
    env->SetLongField(jucx_mem, field, (native_ptr)memh->address);

    return jucx_mem;
}
