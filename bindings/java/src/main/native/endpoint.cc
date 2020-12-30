/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpEndpoint.h"

#include <string.h>    /* memset */


static void error_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    JNIEnv* env = get_jni_env();

    ucs_debug("JUCX: endpoint %p error handler: %s", ep, ucs_status_string(status));
    jobject jucx_ep = reinterpret_cast<jobject>(arg);
    if (env->IsSameObject(jucx_ep, NULL)) {
        ucs_warn("UcpEndpoint was garbage collected. Can't call it's error handler.");
        return;
    }

    jclass jucx_ep_error_hndl_cls = env->FindClass("org/openucx/jucx/ucp/UcpEndpointErrorHandler");
    jclass jucx_ep_class = env->GetObjectClass(jucx_ep);
    jfieldID ep_error_hdnl_field = env->GetFieldID(jucx_ep_class, "errorHandler",
                                                   "Lorg/openucx/jucx/ucp/UcpEndpointErrorHandler;");
    jobject jucx_ep_error_hndl = env->GetObjectField(jucx_ep, ep_error_hdnl_field);
    jmethodID on_error = env->GetMethodID(jucx_ep_error_hndl_cls, "onError",
                                          "(Lorg/openucx/jucx/ucp/UcpEndpoint;ILjava/lang/String;)V");
    jstring error_msg = env->NewStringUTF(ucs_status_string(status));
    env->CallVoidMethod(jucx_ep_error_hndl, on_error, jucx_ep, status, error_msg);
    env->DeleteWeakGlobalRef(jucx_ep);
}


JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_createEndpointNative(JNIEnv *env, jobject jucx_ep,
                                                           jobject ucp_ep_params,
                                                           jlong worker_ptr)
{
    ucp_ep_params_t ep_params;
    jfieldID field;
    ucp_worker_h ucp_worker = (ucp_worker_h)worker_ptr;
    ucp_ep_h endpoint;

    // Get field mask
    jclass ucp_ep_params_class = env->GetObjectClass(ucp_ep_params);
    field = env->GetFieldID(ucp_ep_params_class, "fieldMask", "J");
    ep_params.field_mask = env->GetLongField(ucp_ep_params, field);

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_REMOTE_ADDRESS) {
        field = env->GetFieldID(ucp_ep_params_class, "ucpAddress", "Ljava/nio/ByteBuffer;");
        jobject buf = env->GetObjectField(ucp_ep_params, field);
        ep_params.address = static_cast<const ucp_address_t *>(env->GetDirectBufferAddress(buf));
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) {
        field = env->GetFieldID(ucp_ep_params_class, "errorHandlingMode", "I");
        ep_params.err_mode =  static_cast<ucp_err_handling_mode_t>(env->GetIntField(ucp_ep_params, field));
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_FLAGS) {
        field = env->GetFieldID(ucp_ep_params_class, "flags", "J");
        ep_params.flags = env->GetLongField(ucp_ep_params, field);
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR) {
        struct sockaddr_storage worker_addr;
        socklen_t addrlen;
        memset(&worker_addr, 0, sizeof(struct sockaddr_storage));

        field = env->GetFieldID(ucp_ep_params_class,
                                "socketAddress", "Ljava/net/InetSocketAddress;");
        jobject sock_addr = env->GetObjectField(ucp_ep_params, field);

        if (j2cInetSockAddr(env, sock_addr, worker_addr, addrlen)) {
            ep_params.sockaddr.addr = (const struct sockaddr*)&worker_addr;
            ep_params.sockaddr.addrlen = addrlen;
        }
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_CONN_REQUEST) {
        field = env->GetFieldID(ucp_ep_params_class, "connectionRequest", "J");
        ep_params.conn_request = reinterpret_cast<ucp_conn_request_h>(env->GetLongField(ucp_ep_params, field));
    }

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLER) {
        // Important to use weak reference, to allow JUCX endpoint class to be closed and
        // garbage collected, as error handler may never be called
        ep_params.err_handler.arg = env->NewWeakGlobalRef(jucx_ep);
        ep_params.err_handler.cb = error_handler;
    }

    ucs_status_t status = ucp_ep_create(ucp_worker, &ep_params, &endpoint);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    return (native_ptr)endpoint;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_destroyEndpointNative(JNIEnv *env, jclass cls,
                                                            jlong ep_ptr)
{
    ucp_ep_destroy((ucp_ep_h)ep_ptr);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_closeNonBlockingNative(JNIEnv *env, jclass cls,
                                                             jlong ep_ptr, jint mode)
{
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, NULL, &param);

    if (mode != 0) {
        param.op_attr_mask              |= UCP_OP_ATTR_FIELD_FLAGS;
        param.flags                      = mode;
    }

    param.cb.send                        = jucx_request_callback;

    ucs_status_ptr_t status = ucp_ep_close_nbx((ucp_ep_h)ep_ptr, &param);
    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_unpackRemoteKey(JNIEnv *env, jclass cls,
                                                      jlong ep_ptr, jlong addr)
{
    ucp_rkey_h rkey;

    ucs_status_t status = ucp_ep_rkey_unpack((ucp_ep_h)ep_ptr, (void *)addr, &rkey);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    jobject result = new_rkey_instance(env, rkey);

    /* Coverity thinks that rkey is a leaked object here,
     * but it's stored in a UcpRemoteKey object */
    /* coverity[leaked_storage] */
    return result;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_putNonBlockingNative(JNIEnv *env, jclass cls,
                                                           jlong ep_ptr, jlong laddr,
                                                           jlong size, jlong raddr,
                                                           jlong rkey_ptr, jobject callback,
                                                           jint memory_type)
{
    ucp_request_param_t param;
    jobject jucx_request = jucx_request_allocate(env, callback, &param);

    param.op_attr_mask              |= UCP_OP_ATTR_FIELD_CALLBACK |
                                       UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    param.memory_type                = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.send                    = jucx_request_callback;

    ucs_status_ptr_t status = ucp_put_nbx((ucp_ep_h)ep_ptr, (void *)laddr, size, raddr,
                                          (ucp_rkey_h)rkey_ptr, &param);

    process_request(env, jucx_request, status);

    ucs_trace_req("JUCX: put_nb request %p, of size: %zu, raddr: %zu", status, size, raddr);

    return jucx_request;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_putNonBlockingImplicitNative(JNIEnv *env, jclass cls,
                                                                   jlong ep_ptr, jlong laddr,
                                                                   jlong size, jlong raddr,
                                                                   jlong rkey_ptr)
{
    ucs_status_t status = ucp_put_nbi((ucp_ep_h)ep_ptr, (void *)laddr, size, raddr,
                                      (ucp_rkey_h)rkey_ptr);

    if (UCS_STATUS_IS_ERR(status)) {
        JNU_ThrowExceptionByStatus(env, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_getNonBlockingNative(JNIEnv *env, jclass cls,
                                                           jlong ep_ptr, jlong raddr,
                                                           jlong rkey_ptr, jlong laddr,
                                                           jlong size, jobject callback,
                                                           jint memory_type)
{
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param);

    param.op_attr_mask              |= UCP_OP_ATTR_FIELD_CALLBACK |
                                       UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    param.memory_type                = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.send                    = jucx_request_callback;

    ucs_status_ptr_t status = ucp_get_nbx((ucp_ep_h)ep_ptr, (void *)laddr, size,
                                          raddr, (ucp_rkey_h)rkey_ptr, &param);
    process_request(env, jucx_request, status);
    ucs_trace_req("JUCX: get_nb request %p, raddr: %zu, size: %zu, result address: %zu",
                  status, raddr, size, laddr);

    return jucx_request;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_getNonBlockingImplicitNative(JNIEnv *env, jclass cls,
                                                                   jlong ep_ptr, jlong raddr,
                                                                   jlong rkey_ptr, jlong laddr,
                                                                   jlong size)
{
    ucs_status_t status = ucp_get_nbi((ucp_ep_h)ep_ptr, (void *)laddr, size, raddr,
                                      (ucp_rkey_h)rkey_ptr);

    if (UCS_STATUS_IS_ERR(status)) {
        JNU_ThrowExceptionByStatus(env, status);
    }
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendTaggedNonBlockingNative(JNIEnv *env, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jlong tag,
                                                                  jobject callback, jint memory_type)
                                                                  jint memory_type)
{
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param);

    param.op_attr_mask             |= UCP_OP_ATTR_FIELD_CALLBACK |
                                      UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    param.memory_type               = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.send                   = jucx_request_callback;

    ucs_status_ptr_t status = ucp_tag_send_nbx((ucp_ep_h)ep_ptr, (void *)addr, size, tag, &param);
    process_request(env, jucx_request, status);
    ucs_trace_req("JUCX: send_tag_nb request %p, size: %zu, tag: %ld", status, size, tag);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendTaggedIovNonBlockingNative(JNIEnv *env, jclass cls,
                                                                    jlong ep_ptr, jlongArray addresses,
                                                                    jlongArray sizes, jlong tag,
                                                                    jobject callback, jint memory_type)
{
    int iovcnt;
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param);
    ucp_dt_iov_t* iovec = get_ucp_iov(env, addresses, sizes, iovcnt);
    if (iovec == NULL) {
        return NULL;
    }

    jucx_request_set_iov(env, jucx_request, iovec);

    param.op_attr_mask              |= UCP_OP_ATTR_FIELD_CALLBACK |
                                       UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                                       UCP_OP_ATTR_FIELD_DATATYPE;
    param.memory_type                = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.send                    = jucx_request_callback;
    param.datatype                   = ucp_dt_make_iov();

    ucs_status_ptr_t status = ucp_tag_send_nbx((ucp_ep_h)ep_ptr, iovec, iovcnt, tag, &param);
    process_request(env, jucx_request, status);
    ucs_trace_req("JUCX: send_tag_iov_nb request %p, tag: %ld", status, tag);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendStreamNonBlockingNative(JNIEnv *env, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jobject callback,
                                                                  jint memory_type)
{
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param);

    param.op_attr_mask              |= UCP_OP_ATTR_FIELD_CALLBACK |
                                       UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    param.memory_type                = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.send                    = jucx_request_callback;

    ucs_status_ptr_t status = ucp_stream_send_nbx((ucp_ep_h)ep_ptr, (void *)addr, size, &param);
    process_request(env, jucx_request, status);
    ucs_trace_req("JUCX: send_stream_nb request %p, size: %zu", status, size);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendStreamIovNonBlockingNative(JNIEnv *env, jclass cls,
                                                                     jlong ep_ptr, jlongArray addresses,
                                                                     jlongArray sizes, jobject callback,
                                                                     jint memory_type)
{
    int iovcnt;
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param);
    ucp_dt_iov_t* iovec = get_ucp_iov(env, addresses, sizes, iovcnt);
    if (iovec == NULL) {
        return NULL;
    }

    jucx_request_set_iov(env, jucx_request, iovec);

    param.op_attr_mask              |= UCP_OP_ATTR_FIELD_CALLBACK |
                                       UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                                       UCP_OP_ATTR_FIELD_DATATYPE;
    param.memory_type                = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.send                    = jucx_request_callback;
    param.datatype                   = ucp_dt_make_iov();

    ucs_status_ptr_t status = ucp_stream_send_nbx((ucp_ep_h)ep_ptr, iovec, iovcnt, &param);
    process_request(env, jucx_request, status);
    ucs_trace_req("JUCX: send_stream_iov_nb request %p", status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_recvStreamNonBlockingNative(JNIEnv *env, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jlong flags,
                                                                  jobject callback, jint memory_type)
{
    size_t rlength;
    ucp_request_param_t param;
    jobject jucx_request = jucx_request_allocate(env, callback, &param);

    param.op_attr_mask              |= UCP_OP_ATTR_FIELD_CALLBACK    |
                                       UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                                       UCP_OP_ATTR_FIELD_FLAGS;
    param.memory_type                = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.recv_stream             = stream_recv_callback;
    param.flags                      = flags;

    ucs_status_ptr_t status = ucp_stream_recv_nbx((ucp_ep_h)ep_ptr, (void *)addr, size,
                                                  &rlength, &param);

    if (status == NULL) {
        jucx_request_update_recv_length(env, jucx_request, rlength);
    }

    process_request(env, jucx_request, status);
    ucs_trace_req("JUCX: recv_stream_nb request %p, size: %zu", status, size);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_recvStreamIovNonBlockingNative(JNIEnv *env, jclass cls,
                                                                     jlong ep_ptr,
                                                                     jlongArray addresses, jlongArray sizes,
                                                                     jlong flags, jobject callback,
                                                                     jint memory_type)
{
    size_t rlength;
    int iovcnt;
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param);
    ucp_dt_iov_t* iovec = get_ucp_iov(env, addresses, sizes, iovcnt);
    if (iovec == NULL) {
        return NULL;
    }

    jucx_request_set_iov(env, jucx_request, iovec);

    param.op_attr_mask              |= UCP_OP_ATTR_FIELD_CALLBACK    |
                                       UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                                       UCP_OP_ATTR_FIELD_FLAGS       |
                                       UCP_OP_ATTR_FIELD_DATATYPE;
    param.memory_type                = static_cast<ucs_memory_type_t>(memory_type);
    param.cb.recv_stream             = stream_recv_callback;
    param.datatype                   = ucp_dt_make_iov();
    param.flags                      = flags;

    ucs_status_ptr_t status = ucp_stream_recv_nbx((ucp_ep_h)ep_ptr, iovec, iovcnt, &rlength, &param);

    if (status == NULL) {
        jucx_request_update_recv_length(env, jucx_request, rlength);
    }

    process_request(env, jucx_request, status);
    ucs_trace_req("JUCX: recv_stream_iov_nb request %p", status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_flushNonBlockingNative(JNIEnv *env, jclass cls,
                                                             jlong ep_ptr,
                                                             jobject callback)
{
    ucp_request_param_t param;

    jobject jucx_request = jucx_request_allocate(env, callback, &param);

    param.op_attr_mask      |= UCP_OP_ATTR_FIELD_CALLBACK;
    param.cb.send            = jucx_request_callback;

    ucs_status_ptr_t status = ucp_ep_flush_nbx((ucp_ep_h)ep_ptr, &param);
    process_request(env, jucx_request, status);

    return jucx_request;
}
