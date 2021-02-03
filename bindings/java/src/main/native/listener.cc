/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
#include "org_openucx_jucx_ucp_UcpListener.h"

#include <string.h>    /* memset */


JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_ucp_UcpListener_createUcpListener(JNIEnv *env, jclass cls,
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
        field = env->GetFieldID(jucx_listener_param_class,
                                "connectionHandler", "Lorg/openucx/jucx/ucp/UcpListenerConnectionHandler;");
        jobject jucx_conn_handler = env->GetObjectField(ucp_listener_params, field);
        params.conn_handler.arg = env->NewWeakGlobalRef(jucx_conn_handler);
        params.conn_handler.cb = jucx_connection_handler;
    }

    if (params.field_mask & UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER) {
        field = env->GetFieldID(jucx_listener_param_class,
                                "acceptHandler", "Lorg/openucx/jucx/ucp/UcpListenerAcceptHandler;");
        jobject jucx_accept_hndl = env->GetObjectField(ucp_listener_params, field);
        params.accept_handler.arg = env->NewWeakGlobalRef(jucx_accept_hndl);
        params.accept_handler.cb = jucx_accept_handler;
    }

    ucs_status_t status = ucp_listener_create(ucp_worker, &params, &listener);
    if (status != UCS_OK) {
        JNU_ThrowExceptionByStatus(env, status);
    }

    return (native_ptr)listener;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpListener_destroyUcpListenerNative(JNIEnv *env,
                                                               jclass cls,
                                                               jlong listener_ptr)
{
    ucp_listener_destroy((ucp_listener_h)listener_ptr);
}
