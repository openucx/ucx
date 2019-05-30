/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
extern "C" {
  #include <ucs/arch/cpu.h>
  #include <ucs/debug/assert.h>
  #include <ucs/debug/debug.h>
}

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <pthread.h>   /* pthread_yield */


static JavaVM *jvm_global;

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void* reserved) {
   ucs_debug_disable_signals();
   jvm_global = jvm;
   return JNI_VERSION_1_1;
}

/**
 * Throw a Java exception by name. Similar to SignalError.
 */
JNIEXPORT void JNICALL JNU_ThrowException(JNIEnv *env, const char *msg)
{
    jclass cls = env->FindClass("org/ucx/jucx/UcxException");
    ucs_error("JUCX: %s", msg);
    if (cls != 0) { /* Otherwise an exception has already been thrown */
        env->ThrowNew(cls, msg);
    }
}

void JNU_ThrowExceptionByStatus(JNIEnv *env, ucs_status_t status)
{
    JNU_ThrowException(env, ucs_status_string(status));
}

bool j2cInetSockAddr(JNIEnv *env, jobject sock_addr, sockaddr_storage& ss,  socklen_t& sa_len)
{
    jfieldID field;
    memset(&ss, 0, sizeof(ss));
    sa_len = 0;

    if (sock_addr == NULL) {
        JNU_ThrowException(env, "j2cInetSockAddr: InetSocketAddr is null");
        return false;
    }

    jclass inetsockaddr_cls = env->GetObjectClass(sock_addr);

    // Get sockAddr->port
    jmethodID getPort = env->GetMethodID(inetsockaddr_cls, "getPort", "()I");
    jint port = env->CallIntMethod(sock_addr, getPort);

    // Get sockAddr->getAddress (InetAddress)
    jmethodID getAddress = env->GetMethodID(inetsockaddr_cls, "getAddress",
                                            "()Ljava/net/InetAddress;");
    jobject inet_address = env->CallObjectMethod(sock_addr, getAddress);

    if (inet_address == NULL) {
        JNU_ThrowException(env, "j2cInetSockAddr: InetSocketAddr.getAddress is null");
        return false;
    }

    jclass inetaddr_cls = env->GetObjectClass(inet_address);

    // Get address family. In Java IPv4 has addressFamily = 1, IPv6 = 2.
    field = env->GetFieldID(inetaddr_cls, "holder",
                            "Ljava/net/InetAddress$InetAddressHolder;");
    jobject inet_addr_holder = env->GetObjectField(inet_address, field);
    jclass inet_addr_holder_cls = env->GetObjectClass(inet_addr_holder);
    field = env->GetFieldID(inet_addr_holder_cls, "family", "I");
    jint family = env->GetIntField(inet_addr_holder, field);

    field = env->GetStaticFieldID(inetaddr_cls, "IPv4", "I");
    const int JAVA_IPV4_FAMILY = env->GetStaticIntField(inetaddr_cls, field);
    field = env->GetStaticFieldID(inetaddr_cls, "IPv6", "I");
    const int JAVA_IPV6_FAMILY = env->GetStaticIntField(inetaddr_cls, field);

    // Get the byte array that stores the IP address bytes in the InetAddress.
    jmethodID get_addr_bytes = env->GetMethodID(inetaddr_cls, "getAddress", "()[B");
    jobject ip_byte_array = env->CallObjectMethod(inet_address, get_addr_bytes);

    if (ip_byte_array == NULL) {
        JNU_ThrowException(env, "j2cInetSockAddr: InetAddr.getAddress.getAddress is null");
        return false;
    }

    jbyteArray addressBytes = static_cast<jbyteArray>(ip_byte_array);

    if (family == JAVA_IPV4_FAMILY) {
        // Deal with Inet4Address instances.
        // We should represent this Inet4Address as an IPv4 sockaddr_in.
        ss.ss_family = AF_INET;
        sockaddr_in &sin = reinterpret_cast<sockaddr_in &>(ss);
        sin.sin_port = htons(port);
        jbyte *dst = reinterpret_cast<jbyte *>(&sin.sin_addr.s_addr);
        env->GetByteArrayRegion(addressBytes, 0, 4, dst);
        sa_len = sizeof(sockaddr_in);
        return true;
    } else if (family == JAVA_IPV6_FAMILY) {
        jclass inet6_addr_cls = env->FindClass("java/net/Inet6Address");
        ss.ss_family = AF_INET6;
        sockaddr_in6& sin6 = reinterpret_cast<sockaddr_in6&>(ss);
        sin6.sin6_port = htons(port);
        // IPv6 address. Copy the bytes...
        jbyte *dst = reinterpret_cast<jbyte *>(&sin6.sin6_addr.s6_addr);
        env->GetByteArrayRegion(addressBytes, 0, 16, dst);
        // ...and set the scope id...
        jmethodID getScopeId = env->GetMethodID(inet6_addr_cls, "getScopeId", "()I");
        sin6.sin6_scope_id = env->CallIntMethod(inet_address, getScopeId);
        sa_len = sizeof(sockaddr_in6);
        return true;
    }
    JNU_ThrowException(env, "Unknown InetAddress family");
    return false;
}

void jucx_request_init(void *request)
{
     struct jucx_context *ctx = (struct jucx_context *)request;
     ctx->callback = NULL;
     ctx->jucx_request = NULL;
}

JNIEnv* get_jni_env()
{
    void *env;
    jint rs = jvm_global->AttachCurrentThread(&env, NULL);
    ucs_assert(rs == JNI_OK);
    return (JNIEnv*)env;
}

static inline void set_jucx_request_completed(JNIEnv *env, jobject jucx_request)
{
    jclass jucx_request_cls = env->GetObjectClass(jucx_request);
    jfieldID field = env->GetFieldID(jucx_request_cls, "completed", "Z");
    env->SetBooleanField(jucx_request, field, true);
}

static inline void call_on_success(jobject callback, jobject request)
{
    JNIEnv *env = get_jni_env();
    jclass callback_cls = env->GetObjectClass(callback);
    jmethodID on_success = env->GetMethodID(callback_cls, "onSuccess",
                                            "(Lorg/ucx/jucx/UcxRequest;)V");
    env->CallVoidMethod(callback, on_success, request);
}

static inline void call_on_error(jobject callback, ucs_status_t status)
{
    ucs_error("JUCX: send request error: %s", ucs_status_string(status));
    JNIEnv *env = get_jni_env();
    jclass callback_cls = env->GetObjectClass(callback);
    jmethodID on_error = env->GetMethodID(callback_cls, "onError", "(ILjava/lang/String;)V");
    jstring error_msg = env->NewStringUTF(ucs_status_string(status));
    env->CallVoidMethod(callback, on_error, status, error_msg);
}

void jucx_request_callback(void *request, ucs_status_t status)
{
    struct jucx_context *ctx = (struct jucx_context *)request;
    while (ctx->jucx_request == NULL) {
        pthread_yield();
    }
    ucs_memory_cpu_load_fence();
    JNIEnv *env = get_jni_env();
    set_jucx_request_completed(env, ctx->jucx_request);

    if (ctx->callback != NULL) {
        if (status == UCS_OK) {
            call_on_success(ctx->callback, ctx->jucx_request);
        } else {
            call_on_error(ctx->callback, status);
        }
        env->DeleteGlobalRef(ctx->callback);
    }

    env->DeleteGlobalRef(ctx->jucx_request);
    ctx->callback = NULL;
    ctx->jucx_request = NULL;
    ucp_request_free(request);
}

void recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info)
{
    jucx_request_callback(request, status);
}

jobject process_request(void *request, jobject callback)
{
    JNIEnv *env = get_jni_env();
    jclass jucx_request_cls = env->FindClass("org/ucx/jucx/UcxRequest");
    jmethodID constructor = env->GetMethodID(jucx_request_cls, "<init>", "()V");
    jobject jucx_request = env->NewObject(jucx_request_cls, constructor);

    // If request is a pointer set context callback and rkey.
    if (UCS_PTR_IS_PTR(request)) {
        if (callback != NULL) {
            ((struct jucx_context *)request)->callback = env->NewGlobalRef(callback);
        }
        ucs_memory_cpu_store_fence();
        ((struct jucx_context *)request)->jucx_request = env->NewGlobalRef(jucx_request);
    } else {
        if (UCS_PTR_IS_ERR(request)) {
            JNU_ThrowExceptionByStatus(env, UCS_PTR_STATUS(request));
            if (callback != NULL) {
                call_on_error(callback, UCS_PTR_STATUS(request));
            }
        } else if (UCS_PTR_STATUS(request) == UCS_OK) {
            if (callback != NULL) {
                call_on_success(callback, jucx_request);
            }
        }
        set_jucx_request_completed(env, jucx_request);
    }
    return jucx_request;
}
