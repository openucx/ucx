/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef HELPER_H_
#define HELPER_H_

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>

#include <jni.h>


typedef uintptr_t native_ptr;


JNIEXPORT void JNICALL JNU_ThrowException(JNIEnv *, const char *);

void JNU_ThrowExceptionByStatus(JNIEnv *, ucs_status_t);

#define JUCX_DEFINE_LONG_CONSTANT(_name) do { \
    jfieldID field = env->GetStaticFieldID(cls, #_name, "J"); \
    env->SetStaticLongField(cls, field, _name); \
} while(0)

#define JUCX_DEFINE_INT_CONSTANT(_name) do { \
    jfieldID field = env->GetStaticFieldID(cls, #_name, "I"); \
    env->SetStaticIntField(cls, field, _name); \
} while(0)

/**
 * @brief Utility to convert Java InetSocketAddress class (corresponds to the Network Layer 4
 * and consists of an IP address and a port number) to corresponding sockaddr_storage struct.
 * Supports IPv4 and IPv6.
 */
bool j2cInetSockAddr(JNIEnv *env, jobject sock_addr, sockaddr_storage& ss, socklen_t& sa_len);

struct jucx_context {
    jobject callback;
    volatile jobject jucx_request;
};

void jucx_request_init(void *request);

/**
 * @brief Get the jni env object. To be able to call java methods from ucx async callbacks.
 */
JNIEnv* get_jni_env();

/**
 * @brief Send callback used to invoke java callback class on completion of ucp operations.
 */
void jucx_request_callback(void *request, ucs_status_t status);

/**
 * @brief Recv callback used to invoke java callback class on completion of ucp recv_nb operation.
 */
void recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);

/**
 * @brief Utility to process request logic: if request is pointer - set callback to request context.
 * If request is status - call callback directly.
 * Returns jucx_request object, that could be monitored on completion.
 */
jobject process_request(void *request, jobject callback);

#endif
