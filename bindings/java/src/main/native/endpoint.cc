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
    jstring name = NULL;

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

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_NAME) {
        field = env->GetFieldID(ucp_ep_params_class, "name", "Ljava/lang/String;");
        name = (jstring)env->GetObjectField(ucp_ep_params, field);
        ep_params.name = env->GetStringUTFChars(name, 0);;
    }

    ucs_status_t status = ucp_ep_create(ucp_worker, &ep_params, &endpoint);
    if (name != NULL) {
        env->ReleaseStringChars(name, (const jchar*)ep_params.name);
    }
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
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, NULL, &param, NULL);

    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags         = mode;
    param.cb.send       = jucx_request_callback;

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
                                                           jobject request_params)
{
    ucp_request_param_t param = {0};
    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.cb.send         = jucx_request_callback;

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
                                                           jobject request_params)
{
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.cb.send       = jucx_request_callback;

    ucs_status_ptr_t status = ucp_get_nbx((ucp_ep_h)ep_ptr, (void *)laddr, size,
                                          raddr, (ucp_rkey_h)rkey_ptr, &param);
    ucs_trace_req("JUCX: get_nb request %p, raddr: %zu, size: %zu, result address: %zu",
                  status, raddr, size, laddr);

    process_request(env, jucx_request, status);
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
                                                                  jobject callback,
                                                                  jobject request_params)
{
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.cb.send = jucx_request_callback;

    ucs_status_ptr_t status = ucp_tag_send_nbx((ucp_ep_h)ep_ptr, (void *)addr, size, tag, &param);
    ucs_trace_req("JUCX: send_tag_nb request %p, size: %zu, tag: %ld", status, size, tag);

    process_request(env, jucx_request, status);
    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendTaggedIovNonBlockingNative(JNIEnv *env, jclass cls,
                                                                    jlong ep_ptr, jlongArray addresses,
                                                                    jlongArray sizes, jlong tag,
                                                                    jobject callback,
                                                                    jobject request_params)
{
    int iovcnt;
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);
    ucp_dt_iov_t* iovec = get_ucp_iov(env, addresses, sizes, iovcnt);
    if (iovec == NULL) {
        return NULL;
    }

    jucx_request_set_iov(env, jucx_request, iovec);

    param.op_attr_mask |= UCP_OP_ATTR_FIELD_DATATYPE;
    param.cb.send       = jucx_request_callback;
    param.datatype      = ucp_dt_make_iov();

    ucs_status_ptr_t status = ucp_tag_send_nbx((ucp_ep_h)ep_ptr, iovec, iovcnt, tag, &param);
    ucs_trace_req("JUCX: send_tag_iov_nb request %p, tag: %ld", status, tag);

    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendStreamNonBlockingNative(JNIEnv *env, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jobject callback,
                                                                  jobject request_params)
{
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.cb.send        = jucx_request_callback;

    ucs_status_ptr_t status = ucp_stream_send_nbx((ucp_ep_h)ep_ptr, (void *)addr, size, &param);
    ucs_trace_req("JUCX: send_stream_nb request %p, size: %zu", status, size);

    process_request(env, jucx_request, status);
    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendStreamIovNonBlockingNative(JNIEnv *env, jclass cls,
                                                                     jlong ep_ptr, jlongArray addresses,
                                                                     jlongArray sizes, jobject callback,
                                                                     jobject request_params)
{
    int iovcnt;
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);
    ucp_dt_iov_t* iovec = get_ucp_iov(env, addresses, sizes, iovcnt);
    if (iovec == NULL) {
        return NULL;
    }

    jucx_request_set_iov(env, jucx_request, iovec);

    param.op_attr_mask |= UCP_OP_ATTR_FIELD_DATATYPE;
    param.cb.send       = jucx_request_callback;
    param.datatype      = ucp_dt_make_iov();

    ucs_status_ptr_t status = ucp_stream_send_nbx((ucp_ep_h)ep_ptr, iovec, iovcnt, &param);
    ucs_trace_req("JUCX: send_stream_iov_nb request %p", status);

    process_request(env, jucx_request, status);
    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_recvStreamNonBlockingNative(JNIEnv *env, jclass cls,
                                                                  jlong ep_ptr, jlong addr,
                                                                  jlong size, jlong flags,
                                                                  jobject callback,
                                                                  jobject request_params)
{
    size_t rlength;
    ucp_request_param_t param = {0};
    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.op_attr_mask   |= UCP_OP_ATTR_FIELD_FLAGS;
    param.cb.recv_stream  = stream_recv_callback;
    param.flags           = flags;

    ucs_status_ptr_t status = ucp_stream_recv_nbx((ucp_ep_h)ep_ptr, (void *)addr, size,
                                                  &rlength, &param);
    ucs_trace_req("JUCX: recv_stream_nb request %p, size: %zu", status, size);

    if (status == NULL) {
        jucx_request_update_recv_length(env, jucx_request, rlength);
    }

    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_recvStreamIovNonBlockingNative(JNIEnv *env, jclass cls,
                                                                     jlong ep_ptr,
                                                                     jlongArray addresses, jlongArray sizes,
                                                                     jlong flags, jobject callback,
                                                                     jobject request_params)
{
    size_t rlength;
    int iovcnt;
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);
    ucp_dt_iov_t* iovec = get_ucp_iov(env, addresses, sizes, iovcnt);
    if (iovec == NULL) {
        return NULL;
    }

    jucx_request_set_iov(env, jucx_request, iovec);

    param.op_attr_mask   |= UCP_OP_ATTR_FIELD_FLAGS |
                            UCP_OP_ATTR_FIELD_DATATYPE;
    param.cb.recv_stream  = stream_recv_callback;
    param.datatype        = ucp_dt_make_iov();
    param.flags           = flags;

    ucs_status_ptr_t status = ucp_stream_recv_nbx((ucp_ep_h)ep_ptr, iovec, iovcnt, &rlength,
                                                  &param);
    ucs_trace_req("JUCX: recv_stream_iov_nb request %p", status);

    if (status == NULL) {
        jucx_request_update_recv_length(env, jucx_request, rlength);
    }

    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_flushNonBlockingNative(JNIEnv *env, jclass cls,
                                                             jlong ep_ptr, jobject callback)
{
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, NULL);

    param.cb.send = jucx_request_callback;

    ucs_status_ptr_t status = ucp_ep_flush_nbx((ucp_ep_h)ep_ptr, &param);
    ucs_trace_req("JUCX: ucp_ep_flush_nbx request %p", status);

    process_request(env, jucx_request, status);

    return jucx_request;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_sendAmNonBlockingNative(JNIEnv *env, jclass cls,
                                                              jlong ep_ptr, jint am_id,
                                                              jlong header_addr, jlong header_length,
                                                              jlong data_address, jlong data_length,
                                                              jlong flags, jobject callback,
                                                              jobject request_params)
{
    ucp_request_param_t param = {0};

    jobject jucx_request = jucx_request_allocate(env, callback, &param, request_params);

    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.cb.send       = jucx_request_callback;
    param.flags         = flags;

    ucs_status_ptr_t status = ucp_am_send_nbx((ucp_ep_h)ep_ptr, am_id, (void*)header_addr, header_length,
                                              (void*)data_address, data_length, &param);
    ucs_trace_req("JUCX: ucp_am_send_nbx request %p", status);

    process_request(env, jucx_request, status);
    return jucx_request;
}
