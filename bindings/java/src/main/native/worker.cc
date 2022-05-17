/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpWorker.h"

/**
 * Bridge method for creating ucp_worker from java
 */
JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_createWorkerNative(JNIEnv *env, jobject jucx_worker,
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

    if (worker_params.field_mask & UCP_WORKER_PARAM_FIELD_CLIENT_ID) {
        field = env->GetFieldID(jucx_param_class, "clientId", "J");
        worker_params.client_id = env->GetLongField(jucx_worker_params, field);
    }

    ucs_status_t status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
        return -1L;
    }

    ucp_worker_attr_t attr = {0};
    attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;

    status = ucp_worker_query(ucp_worker, &attr);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    field = env->GetFieldID(env->GetObjectClass(jucx_worker), "maxAmHeaderSize", "J");
    env->SetLongField(jucx_worker, field, attr.max_am_header);

    return (native_ptr)ucp_worker;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_releaseWorkerNative(JNIEnv *env, jclass cls,
                                                        jlong ucp_worker_ptr)
{
    ucp_worker_destroy((ucp_worker_h)ucp_worker_ptr);
}


JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_workerGetAddressNative(JNIEnv *env, jclass cls,
                                                           jlong ucp_worker_ptr)
{
    ucp_address_t *addr;
    size_t len;
    ucs_status_t status;

    status = ucp_worker_get_address((ucp_worker_h)ucp_worker_ptr, &addr, &len);

    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
        return NULL;
    }

    return env->NewDirectByteBuffer(addr, len);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_releaseAddressNative(JNIEnv *env, jclass cls,
                                                         jlong ucp_worker_ptr,
                                                         jobject ucp_address)
{

    ucp_worker_release_address((ucp_worker_h)ucp_worker_ptr,
                               (ucp_address_t *)env->GetDirectBufferAddress(ucp_address));
}

JNIEXPORT jint JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_progressWorkerNative(JNIEnv *env, jclass cls, jlong ucp_worker_ptr)
{
    return ucp_worker_progress((ucp_worker_h)ucp_worker_ptr);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_flushNonBlockingNative(JNIEnv *env, jclass cls,
                                                           jlong ucp_worker_ptr,
                                                           jobject callback)
{
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param, NULL);

    param.cb.send = jucx_request_callback;

    ucs_status_ptr_t status = ucp_worker_flush_nbx((ucp_worker_h)ucp_worker_ptr, &param);
    ucs_trace_req("JUCX: ucp_worker_flush_nbx request %p", status);

    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_waitWorkerNative(JNIEnv *env, jclass cls, jlong ucp_worker_ptr)
{
    ucs_status_t status = ucp_worker_wait((ucp_worker_h)ucp_worker_ptr);

    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_signalWorkerNative(JNIEnv *env, jclass cls, jlong ucp_worker_ptr)
{
    ucs_status_t status = ucp_worker_signal((ucp_worker_h)ucp_worker_ptr);

    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_recvTaggedNonBlockingNative(JNIEnv *env, jclass cls,
                                                                jlong ucp_worker_ptr,
                                                                jlong laddr, jlong size,
                                                                jlong tag, jlong tag_mask,
                                                                jobject callback,
                                                                jobject request_params)
{
    ucp_request_param_t param = {0};
    ucp_tag_recv_info_t recv_info = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.op_attr_mask       |= UCP_OP_ATTR_FIELD_RECV_INFO;
    param.cb.recv             = recv_callback;
    param.recv_info.tag_info  = &recv_info;

    ucs_status_ptr_t status = ucp_tag_recv_nbx((ucp_worker_h)ucp_worker_ptr,
                                                (void *)laddr, size, tag, tag_mask, &param);
    ucs_trace_req("JUCX: tag_recv_nb request %p, msg size: %zu, tag: %ld", status, size, tag);

    if (UCS_PTR_STATUS(status) == UCS_OK) {
        jucx_request_update_recv_length(env, jucx_request, recv_info.length);
        jucx_request_update_sender_tag(env, jucx_request, recv_info.sender_tag);
    }

    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_recvTaggedIovNonBlockingNative(JNIEnv *env, jclass cls,
                                                                   jlong ucp_worker_ptr,
                                                                   jlongArray addresses,
                                                                   jlongArray sizes, jlong tag,
                                                                   jlong tag_mask, jobject callback,
                                                                   jobject request_params)
{
    int iovcnt;
    ucp_request_param_t param = {0};
    ucp_tag_recv_info_t recv_info = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);
    ucp_dt_iov_t* iovec = get_ucp_iov(env, addresses, sizes, iovcnt);
    if (iovec == NULL) {
        return NULL;
    }

    jucx_request_set_iov(env, jucx_request, iovec);

    param.op_attr_mask       |= UCP_OP_ATTR_FIELD_RECV_INFO |
                                UCP_OP_ATTR_FIELD_DATATYPE;
    param.cb.recv             = recv_callback;
    param.datatype            = ucp_dt_make_iov();
    param.recv_info.tag_info  = &recv_info;

    ucs_status_ptr_t status = ucp_tag_recv_nbx((ucp_worker_h)ucp_worker_ptr,
                                               iovec, iovcnt, tag, tag_mask, &param);
    ucs_trace_req("JUCX: tag_recv_iov_nb request %p, tag: %ld", status, tag);

    if (UCS_PTR_STATUS(status) == UCS_OK) {
        jucx_request_update_recv_length(env, jucx_request, recv_info.length);
        jucx_request_update_sender_tag(env, jucx_request, recv_info.sender_tag);
    }
    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_tagProbeNonBlockingNative(JNIEnv *env, jclass cls,
                                                              jlong ucp_worker_ptr,
                                                              jlong tag, jlong tag_mask,
                                                              jboolean remove)
{
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag = ucp_tag_probe_nb((ucp_worker_h)ucp_worker_ptr, tag, tag_mask,
                                                 remove, &info_tag);
    jobject result = NULL;

    if (msg_tag != NULL) {
        result = new_tag_msg_instance(env, msg_tag, &info_tag);
    }

    return result;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_recvTaggedMessageNonBlockingNative(JNIEnv *env, jclass cls,
                                                                       jlong ucp_worker_ptr,
                                                                       jlong laddr, jlong size,
                                                                       jlong msg_ptr,
                                                                       jobject callback,
                                                                       jobject request_params)
{
    ucp_request_param_t param = {0};
    ucp_tag_recv_info_t recv_info = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.op_attr_mask       |= UCP_OP_ATTR_FIELD_RECV_INFO;
    param.cb.recv             = recv_callback;
    param.recv_info.tag_info  = &recv_info;

    ucs_status_ptr_t status = ucp_tag_msg_recv_nbx((ucp_worker_h)ucp_worker_ptr,
                                                   (void *)laddr, size,
                                                   (ucp_tag_message_h)msg_ptr,
                                                   &param);
    ucs_trace_req("JUCX: tag_msg_recv_nb request %p, msg size: %zu, msg: %p", status, size,
                  (ucp_tag_message_h)msg_ptr);

    if (UCS_PTR_STATUS(status) == UCS_OK) {
        jucx_request_update_recv_length(env, jucx_request, recv_info.length);
        jucx_request_update_sender_tag(env, jucx_request, recv_info.sender_tag);
    }

    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_cancelRequestNative(JNIEnv *env, jclass cls,
                                                        jlong ucp_worker_ptr,
                                                        jlong ucp_request_ptr)
{
    ucp_request_cancel((ucp_worker_h)ucp_worker_ptr, (void *)ucp_request_ptr);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_setAmRecvHandlerNative(JNIEnv *env, jclass cls,
                                                           jlong ucp_worker_ptr, jint amId,
                                                           jobjectArray callbackAndWorker,
                                                           jlong flags)
{
    ucp_am_handler_param_t param = {0};
    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID    |
                       UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                       UCP_AM_HANDLER_PARAM_FIELD_CB    |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = amId;
    param.flags      = flags;
    if (callbackAndWorker == NULL) {
        param.cb     = NULL;
    } else {
        param.cb     = am_recv_callback;
        param.arg    = env->NewWeakGlobalRef(callbackAndWorker);
    }

    ucs_status_t status = ucp_worker_set_am_recv_handler((ucp_worker_h)ucp_worker_ptr, &param);

    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_recvAmDataNonBlockingNative(JNIEnv *env, jclass cls,
                                                                jlong ucp_worker_ptr,
                                                                jlong data_descr_ptr,
                                                                jlong address, jlong length,
                                                                jobject callback,
                                                                jobject request_params)
{
    ucp_request_param_t param = {0};
    size_t recv_length;


    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.op_attr_mask     |= UCP_OP_ATTR_FIELD_RECV_INFO;
    param.cb.recv_am        = stream_recv_callback;
    param.recv_info.length  = &recv_length;

    ucs_status_ptr_t status = ucp_am_recv_data_nbx((ucp_worker_h)ucp_worker_ptr, (void*)data_descr_ptr,
                                                   (void*)address, length, &param);
    ucs_trace_req("JUCX: ucp_am_recv_data_nbx request %p, msg size: %zu, data: %p", status, length,
                  (void*)data_descr_ptr);

    if (UCS_PTR_STATUS(status) == UCS_OK) {
        jucx_request_update_recv_length(env, jucx_request, recv_length);
    }

    process_request(env, jucx_request, status);
    return jucx_request;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpWorker_amDataReleaseNative(JNIEnv *env, jclass cls,
                                                        jlong ucp_worker_ptr, jlong data_descr_ptr)
{
    ucp_am_data_release((ucp_worker_h)ucp_worker_ptr, (void*)data_descr_ptr);
}
