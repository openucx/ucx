/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpConnectionRequest.h"
#include "org_openucx_jucx_ucp_UcpListener.h"

#include <string.h>    /* memset */


JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_ucp_UcpListener_createUcpListener(JNIEnv *env, jobject listener_obj,
                                                        jobject ucp_listener_params,
                                                        jlong worker_ptr)
{
    ucp_listener_params_t params;
    ucp_listener_h listener;
    jfieldID field;
    ucp_worker_h ucp_worker = (ucp_worker_h)worker_ptr;

    // Get field mask
    jclass jucx_listener_param_class = env->GetObjectClass(ucp_listener_params);
    field = env->GetFieldID(jucx_listener_param_class, "fieldMask", "J");
    params.field_mask = env->GetLongField(ucp_listener_params, field);

    // Get sockAddr
    field = env->GetFieldID(jucx_listener_param_class,
                            "sockAddr", "Ljava/net/InetSocketAddress;");
    jobject sock_addr = env->GetObjectField(ucp_listener_params, field);

    struct sockaddr_storage listen_addr;
    socklen_t addrlen;
    memset(&listen_addr, 0, sizeof(struct sockaddr_storage));

    if (!j2cInetSockAddr(env, sock_addr, listen_addr, addrlen)) {
        return -1;
    }

    params.sockaddr.addr = (const struct sockaddr*)&listen_addr;
    params.sockaddr.addrlen = addrlen;

    if (params.field_mask & UCP_LISTENER_PARAM_FIELD_CONN_HANDLER) {
        params.conn_handler.arg = env->NewWeakGlobalRef(listener_obj);
        params.conn_handler.cb = jucx_connection_handler;
    }

    ucs_status_t status = ucp_listener_create(ucp_worker, &params, &listener);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    return (native_ptr)listener;
}

JNIEXPORT jobject JNICALL
Java_org_openucx_jucx_ucp_UcpListener_queryAddressNative(JNIEnv *env,
                                                         jclass cls,
                                                         jlong listener_ptr)
{
    ucp_listener_attr_t listener_attr;
    listener_attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;

    ucs_status_t status = ucp_listener_query((ucp_listener_h)listener_ptr, &listener_attr);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    return c2jInetSockAddr(env, &listener_attr.sockaddr);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpListener_destroyUcpListenerNative(JNIEnv *env,
                                                               jclass cls,
                                                               jlong listener_ptr)
{
    ucp_listener_destroy((ucp_listener_h)listener_ptr);
}


JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpConnectionRequest_rejectConnRequestNative(JNIEnv *env, jclass cls,
                                                                       jlong listener_ptr,
                                                                       jlong conn_request_ptr)
{
    ucs_status_t status = ucp_listener_reject((ucp_listener_h)listener_ptr,
                                              (ucp_conn_request_h)conn_request_ptr);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }
}
