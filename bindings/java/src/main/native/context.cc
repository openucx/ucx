/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpContext.h"
extern "C" {
#include <ucp/core/ucp_mm.h>
}

/**
 * Bridge method for creating ucp_context from java
 */
JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_ucp_UcpContext_createContextNative(JNIEnv *env, jclass cls,
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
Java_org_openucx_jucx_ucp_UcpContext_cleanupContextNative(JNIEnv *env, jclass cls,
                                                          jlong ucp_context_ptr)
{
    ucp_cleanup((ucp_context_h)ucp_context_ptr);
}


JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpContext_memoryMapNative(JNIEnv *env, jobject ctx,
                                                     jlong ucp_context_ptr,
                                                     jobject jucx_mmap_params)
{
    ucp_mem_map_params_t params = {0};
    ucp_mem_h memh;
    jfieldID field;

    jclass jucx_mmap_class = env->GetObjectClass(jucx_mmap_params);
    field = env->GetFieldID(jucx_mmap_class, "fieldMask", "J");
    params.field_mask = env->GetLongField(jucx_mmap_params, field);

    if (params.field_mask & UCP_MEM_MAP_PARAM_FIELD_ADDRESS) {
        field = env->GetFieldID(jucx_mmap_class, "address", "J");
        params.address = (void *)env->GetLongField(jucx_mmap_params, field);;
    }

    if (params.field_mask & UCP_MEM_MAP_PARAM_FIELD_LENGTH) {
        field = env->GetFieldID(jucx_mmap_class, "length", "J");
        params.length = env->GetLongField(jucx_mmap_params, field);;
    }

    if (params.field_mask & UCP_MEM_MAP_PARAM_FIELD_FLAGS) {
        field = env->GetFieldID(jucx_mmap_class, "flags", "J");
        params.flags = env->GetLongField(jucx_mmap_params, field);;
    }

    ucs_status_t status =  ucp_mem_map((ucp_context_h)ucp_context_ptr, &params, &memh);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    // Construct UcpMemory class
    jclass jucx_mem_cls = env->FindClass("org/openucx/jucx/ucp/UcpMemory");
    jmethodID constructor = env->GetMethodID(jucx_mem_cls, "<init>", "(J)V");
    jobject jucx_mem = env->NewObject(jucx_mem_cls, constructor, (native_ptr)memh);

    // Set UcpContext pointer
    field = env->GetFieldID(jucx_mem_cls, "context", "Lorg/openucx/jucx/ucp/UcpContext;");
    env->SetObjectField(jucx_mem, field, ctx);

    // Set address
    field =  env->GetFieldID(jucx_mem_cls, "address", "J");
    env->SetLongField(jucx_mem, field, (native_ptr)memh->address);

    // Set length
    field =  env->GetFieldID(jucx_mem_cls, "length", "J");
    env->SetLongField(jucx_mem, field, memh->length);

    /* Coverity thinks that memh is a leaked object here,
     * but it's stored in a UcpMemory object */
    /* coverity[leaked_storage] */
    return jucx_mem;
}
