/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <ucp/api/ucp.h>
#include "helper.h"

#include "org_ucx_jucx_Bridge.h"

/**
 * Bridge methods for creating ucp_worker from java
 */
JNIEXPORT jlong JNICALL
Java_org_ucx_jucx_Bridge_createWorkerNative(JNIEnv *env, jclass cls, jobject jucx_worker_params,
                                            jlong context_ptr)
{
    ucp_worker_params_t worker_params = { 0 };
    jfieldID field;
    ucp_worker_h ucp_worker;
    ucp_context_h ucp_context = (ucp_context_h)context_ptr;
    jclass jucx_param_class = env->GetObjectClass(jucx_worker_params);

    field = env->GetFieldID(jucx_param_class, "fieldMask", "J");
    worker_params.field_mask = env->GetLongField(jucx_worker_params, field);

    if (worker_params.field_mask & UCP_WORKER_PARAM_FIELD_THREAD_MODE) {
        field = env->GetFieldID(jucx_param_class, "threadMode", "Lorg/ucx/jucx/UcxTools$UcsTreadMode;");
        jobject thread_mode = env->GetObjectField(jucx_worker_params, field);
        jclass thread_mode_class = env->GetObjectClass(thread_mode);
        jmethodID ordinal_method_id = env->GetMethodID(thread_mode_class, "ordinal", "()I");
        worker_params.thread_mode = static_cast<ucs_thread_mode_t>(env->CallIntMethod(jucx_worker_params, ordinal_method_id));
    }

    if (worker_params.field_mask & UCP_WORKER_PARAM_FIELD_CPU_MASK) {
        ucs_cpu_set_t cpu_mask;
        UCS_CPU_ZERO(&cpu_mask);
        field = env->GetFieldID(jucx_param_class, "cpuMask", "Ljava/util/BitSet;");
        jobject cpu_mask_bitset = env->GetObjectField(jucx_worker_params, field);
        jclass bitset_class = env->FindClass("java/util/BitSet");
        jmethodID next_set_bit = env->GetMethodID(bitset_class, "nextSetBit", "(I)I");
        for (jint bit_index = env->CallIntMethod(cpu_mask_bitset, next_set_bit, 0); bit_index >=0;
                  bit_index = env->CallIntMethod(cpu_mask_bitset, next_set_bit, bit_index + 1)) {
            UCS_CPU_SET(bit_index, &cpu_mask);
        }
        worker_params.cpu_mask = cpu_mask;
    }

    if (worker_params.field_mask & UCP_WORKER_PARAM_FIELD_EVENTS) {
        field = env->GetFieldID(jucx_param_class, "events", "J");
        worker_params.events = env->GetLongField(jucx_worker_params, field);
    }

    if (worker_params.field_mask & UCP_WORKER_PARAM_FIELD_USER_DATA) {
        field = env->GetFieldID(jucx_param_class, "userData", "Ljava/nio/ByteBuffer;");
        jobject user_data = env->GetObjectField(jucx_worker_params, field);
        worker_params.user_data = env->GetDirectBufferAddress(user_data);
    }

    if (worker_params.field_mask & UCP_WORKER_PARAM_FIELD_EVENT_FD) {
        field = env->GetFieldID(jucx_param_class, "eventFD", "I");
        worker_params.event_fd = env->GetIntField(jucx_worker_params, field);
    }

    ucs_status_t status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
    return (native_ptr)ucp_worker;
}


JNIEXPORT void
JNICALL Java_org_ucx_jucx_Bridge_releaseWorkerNative(JNIEnv *env, jclass cls, jlong ucp_worker_ptr)
{
    ucp_worker_destroy((ucp_worker_h)ucp_worker_ptr);
}