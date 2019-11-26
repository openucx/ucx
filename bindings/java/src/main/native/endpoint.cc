/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpEndpoint.h"

#include <string.h>    /* memset */

#include <ucp/core/ucp_ep.inl> /* ucp_ep_peer_name */


static void error_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    JNIEnv* env = get_jni_env();
    JNU_ThrowExceptionByStatus(env, status);
    ucs_error("JUCX: endpoint error handler: %s", ucs_status_string(status));
}

JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_createEndpointNative(JNIEnv *env, jclass cls,
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

    if (ep_params.field_mask & UCP_EP_PARAM_FIELD_USER_DATA) {
        field = env->GetFieldID(ucp_ep_params_class, "userData", "Ljava/nio/ByteBuffer;");
        jobject user_data = env->GetObjectField(ucp_ep_params, field);
        ep_params.user_data = env->GetDirectBufferAddress(user_data);
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

    ep_params.field_mask |= UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.err_handler.cb = error_handler;

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
    ucp_ep_destroy((ucp_ep_h) ep_ptr);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_closeNonBlockingNative(JNIEnv *env, jclass cls,
                                                             jlong ep_ptr, jint mode)
{
    ucs_status_ptr_t request = ucp_ep_close_nb((ucp_ep_h) ep_ptr, mode);

    return process_request(request, NULL);
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

    jclass ucp_rkey_cls = env->FindClass("org/openucx/jucx/ucp/UcpRemoteKey");
    jmethodID constructor = env->GetMethodID(ucp_rkey_cls, "<init>", "(J)V");
    jobject result = env->NewObject(ucp_rkey_cls, constructor, (native_ptr)rkey);

    /* Coverity thinks that rkey is a leaked object here,
     * but it's stored in a UcpRemoteKey object */
    /* coverity[leaked_storage] */
    return result;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_putNonBlockingNative(JNIEnv *env, jclass cls,
                                                           jlong ep_ptr, jlong laddr,
                                                           jlong size, jlong raddr,
                                                           jlong rkey_ptr, jobject callback)
{
    ucs_status_ptr_t request = ucp_put_nb((ucp_ep_h)ep_ptr, (void *)laddr, size, raddr,
                                          (ucp_rkey_h)rkey_ptr, jucx_request_callback);

    ucs_trace_req("JUCX: put_nb request %p to %s, of size: %zu, raddr: %zu",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), size, raddr);
    return process_request(request, callback);
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
                                                           jlong size, jobject callback)
{
    ucs_status_ptr_t request = ucp_get_nb((ucp_ep_h)ep_ptr, (void *)laddr, size,
                                          raddr, (ucp_rkey_h)rkey_ptr, jucx_request_callback);

    ucs_trace_req("JUCX: get_nb request %p to %s, raddr: %zu, size: %zu, result address: %zu",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), raddr, size, laddr);
    return process_request(request, callback);
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
                                                                  jobject callback)
{
    ucs_status_ptr_t request = ucp_tag_send_nb((ucp_ep_h)ep_ptr, (void *)addr, size,
                                               ucp_dt_make_contig(1), tag, jucx_request_callback);

    ucs_trace_req("JUCX: send_nb request %p to %s, size: %zu, tag: %ld",
                  request, ucp_ep_peer_name((ucp_ep_h)ep_ptr), size, tag);
    return process_request(request, callback);
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpEndpoint_flushNonBlockingNative(JNIEnv *env, jclass cls,
                                                             jlong ep_ptr,
                                                             jobject callback)
{
    ucs_status_ptr_t request = ucp_ep_flush_nb((ucp_ep_h)ep_ptr, 0, jucx_request_callback);

    return process_request(request, callback);
}
