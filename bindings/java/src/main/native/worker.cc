/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_ucx_jucx_ucp_UcpWorker.h"

/**
 * Bridge method for creating ucp_worker from java
 */
JNIEXPORT jlong JNICALL
Java_org_ucx_jucx_ucp_UcpWorker_createWorkerNative(JNIEnv *env, jclass cls,
                                                   jobject jucx_worker_params,
                                                   jlong context_ptr)
{
    ucp_worker_params_t worker_params = { 0 };
    ucp_worker_h ucp_worker;
    ucp_context_h ucp_context = (ucp_context_h)context_ptr;
    jfieldID field;

    jclass jucx_param_class = env->GetObjectClass(jucx_worker_params);
    field = env->GetFieldID(jucx_param_class, "fieldMask", "J");
    worker_params.field_mask = env->GetLongField(jucx_worker_params, field);

    if (worker_params.field_mask & UCP_WORKER_PARAM_FIELD_THREAD_MODE) {
        field = env->GetFieldID(jucx_param_class, "threadMode", "I");
        worker_params.thread_mode = static_cast<ucs_thread_mode_t>(
            env->GetIntField(jucx_worker_params, field));
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

JNIEXPORT void JNICALL
Java_org_ucx_jucx_ucp_UcpWorker_releaseWorkerNative(JNIEnv *env, jclass cls,
                                                    jlong ucp_worker_ptr)
{
    ucp_worker_destroy((ucp_worker_h)ucp_worker_ptr);
}
