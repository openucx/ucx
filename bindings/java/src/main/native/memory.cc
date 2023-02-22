/*
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpMemory.h"
#include "org_openucx_jucx_ucp_UcpRemoteKey.h"
#include "org_openucx_jucx_UcxUtils.h"


JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_unmapMemoryNative(JNIEnv *env, jclass cls,
                                                      jlong context_ptr, jlong mem_ptr)
{
    ucs_status_t status = ucp_mem_unmap((ucp_context_h)context_ptr, (ucp_mem_h)mem_ptr);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_getRkeyBufferNative(JNIEnv *env, jclass cls,
                                                        jlong context_ptr, jlong mem_ptr)
{
    void *rkey_buffer;
    size_t rkey_size;

    ucs_status_t status = ucp_rkey_pack((ucp_context_h)context_ptr, (ucp_mem_h)mem_ptr,
                                        &rkey_buffer, &rkey_size);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
    return env->NewDirectByteBuffer(rkey_buffer, rkey_size);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_releaseRkeyBufferNative(JNIEnv *env, jclass cls, jobject rkey_buf)
{
    ucp_rkey_buffer_release(env->GetDirectBufferAddress(rkey_buf));
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpRemoteKey_rkeyDestroy(JNIEnv *env, jclass cls, jlong rkey_ptr)
{
    ucp_rkey_destroy((ucp_rkey_h) rkey_ptr);
}

JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_UcxUtils_getAddressNative(JNIEnv *env, jclass cls, jobject buffer)
{
    return (native_ptr)env->GetDirectBufferAddress(buffer);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_UcxUtils_getByteBufferViewNative(JNIEnv *env, jclass cls,
                                                       jlong address, jlong size)
{
    return env->NewDirectByteBuffer((void*)address, size);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_getMkeyBufferNative(JNIEnv *env, jclass cls,
                                                        jlong mem_ptr,
                                                        jobject jucx_mem_pack_params)
{
    ucp_memh_pack_params_t params = {};
    void *memh_buffer = NULL;
    size_t memh_size = 0;
    jfieldID field;

    jclass jucx_pack_params_class = env->GetObjectClass(jucx_mem_pack_params);
    field = env->GetFieldID(jucx_pack_params_class, "fieldMask", "J");
    params.field_mask = env->GetLongField(jucx_mem_pack_params, field);

    if (params.field_mask & UCP_MEMH_PACK_PARAM_FIELD_FLAGS) {
        field = env->GetFieldID(jucx_pack_params_class, "flags", "J");
        params.flags = env->GetLongField(jucx_mem_pack_params, field);
    }

    ucs_status_t status = ucp_memh_pack((ucp_mem_h)mem_ptr, &params,
                                        &memh_buffer, &memh_size);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    return env->NewDirectByteBuffer(memh_buffer, memh_size);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpMemory_releaseMkeyBufferNative(JNIEnv *env, jclass cls,
                                                            jobject memh_buf)
{
    ucp_memh_buffer_release_params_t params = {};

    ucp_memh_buffer_release(env->GetDirectBufferAddress(memh_buf), &params);
}

