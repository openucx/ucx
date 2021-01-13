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

#include <arpa/inet.h> /* inet_addr */
#include <locale.h>    /* setlocale */
#include <string.h>    /* memset */


static JavaVM *jvm_global;
static jclass jucx_request_cls;
static jfieldID native_id_field;
static jfieldID recv_size_field;
static jfieldID sender_tag_field;
static jfieldID request_status;
static jmethodID on_success;
static jmethodID jucx_request_constructor;
static jclass ucp_rkey_cls;
static jmethodID ucp_rkey_cls_constructor;
static jclass ucp_tag_msg_cls;
static jmethodID ucp_tag_msg_cls_constructor;

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void* reserved) {
    setlocale(LC_NUMERIC, "C");
    ucs_debug_disable_signals();
    jvm_global = jvm;
    JNIEnv* env;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_1) != JNI_OK) {
       return JNI_ERR;
    }

    jclass jucx_request_cls_local = env->FindClass("org/openucx/jucx/ucp/UcpRequest");
    jucx_request_cls = (jclass) env->NewGlobalRef(jucx_request_cls_local);
    jclass jucx_callback_cls = env->FindClass("org/openucx/jucx/UcxCallback");
    native_id_field = env->GetFieldID(jucx_request_cls, "nativeId", "Ljava/lang/Long;");
    request_status = env->GetFieldID(jucx_request_cls, "status", "I");
    recv_size_field = env->GetFieldID(jucx_request_cls, "recvSize", "J");
    sender_tag_field = env->GetFieldID(jucx_request_cls, "senderTag", "J");
    on_success = env->GetMethodID(jucx_callback_cls, "onSuccess",
                                  "(Lorg/openucx/jucx/ucp/UcpRequest;)V");
    jucx_request_constructor = env->GetMethodID(jucx_request_cls, "<init>", "(J)V");

    jclass ucp_rkey_cls_local = env->FindClass("org/openucx/jucx/ucp/UcpRemoteKey");
    ucp_rkey_cls = (jclass) env->NewGlobalRef(ucp_rkey_cls_local);
    ucp_rkey_cls_constructor = env->GetMethodID(ucp_rkey_cls, "<init>", "(J)V");
    jclass ucp_tag_msg_cls_local = env->FindClass("org/openucx/jucx/ucp/UcpTagMessage");
    ucp_tag_msg_cls = (jclass) env->NewGlobalRef(ucp_tag_msg_cls_local);
    ucp_tag_msg_cls_constructor = env->GetMethodID(ucp_tag_msg_cls, "<init>", "(JJJ)V");
    return JNI_VERSION_1_1;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *jvm, void *reserved) {
    JNIEnv* env;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_1) != JNI_OK) {
        return;
    }

    if (jucx_request_cls != NULL) {
        env->DeleteGlobalRef(jucx_request_cls);
    }
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

static inline void jucx_context_reset(struct jucx_context* ctx)
{
    ctx->callback = NULL;
    ctx->jucx_request = NULL;
    ctx->status = UCS_INPROGRESS;
    ctx->length = 0;
    ctx->iovec = NULL;
    ctx->sender_tag = 0;
}

void jucx_request_init(void *request)
{
     struct jucx_context *ctx = (struct jucx_context *)request;
     jucx_context_reset(ctx);
     ucs_recursive_spinlock_init(&ctx->lock, 0);
}

JNIEnv* get_jni_env()
{
    void *env;
    jint rs = jvm_global->AttachCurrentThread(&env, NULL);
    ucs_assert_always(rs == JNI_OK);
    return (JNIEnv*)env;
}

void jucx_request_update_status(JNIEnv *env, jobject jucx_request, ucs_status_t status)
{
    env->SetIntField(jucx_request, request_status, status);
}

static inline void set_jucx_request_completed(JNIEnv *env, jobject jucx_request,
                                              struct jucx_context *ctx, ucs_status_t status)
{
    env->SetObjectField(jucx_request, native_id_field, NULL);
    jucx_request_update_status(env, jucx_request, status);
    if (ctx != NULL) {
        /* sender_tag and length are initialized to 0,
         * so try to avoid the overhead of setting them again */
        if (ctx->sender_tag != 0) {
            env->SetLongField(jucx_request, sender_tag_field, ctx->sender_tag);
        }

        if (ctx->length > 0) {
            env->SetLongField(jucx_request, recv_size_field, ctx->length);
        }

        if (ctx->iovec != NULL) {
            ucs_free(ctx->iovec);
        }
    }
}

static inline void call_on_success(jobject callback, jobject request)
{
    JNIEnv *env = get_jni_env();
    env->CallVoidMethod(callback, on_success, request);
}

static inline void call_on_error(jobject callback, ucs_status_t status)
{
    if (status == UCS_ERR_CANCELED) {
        ucs_debug("JUCX: Request canceled");
    } else {
        ucs_error("JUCX: request error: %s", ucs_status_string(status));
    }

    JNIEnv *env = get_jni_env();
    jclass callback_cls = env->GetObjectClass(callback);
    jmethodID on_error = env->GetMethodID(callback_cls, "onError", "(ILjava/lang/String;)V");
    jstring error_msg = env->NewStringUTF(ucs_status_string(status));
    env->CallVoidMethod(callback, on_error, status, error_msg);
}

static inline void jucx_call_callback(jobject callback, jobject jucx_request,
                                      ucs_status_t status)
{
    if (status == UCS_OK) {
        UCS_PROFILE_CALL_VOID(call_on_success, callback, jucx_request);
    } else {
        call_on_error(callback, status);
    }
}

UCS_PROFILE_FUNC_VOID(jucx_request_callback, (request, status), void *request, ucs_status_t status)
{
    struct jucx_context *ctx = (struct jucx_context *)request;
    ucs_recursive_spin_lock(&ctx->lock);
    if (ctx->jucx_request == NULL) {
        // here because 1 of 2 reasons:
        // 1. progress is in another thread and got here earlier then process_request happened.
        // 2. this callback is inside ucp_tag_recv_nb function.
        ctx->status = status;
        ucs_recursive_spin_unlock(&ctx->lock);
        return;
    }

    JNIEnv *env = get_jni_env();
    set_jucx_request_completed(env, ctx->jucx_request, ctx, status);

    if (ctx->callback != NULL) {
        jucx_call_callback(ctx->callback, ctx->jucx_request, status);
        env->DeleteGlobalRef(ctx->callback);
    }

    env->DeleteGlobalRef(ctx->jucx_request);
    jucx_context_reset(ctx);
    ucp_request_free(request);
    ucs_recursive_spin_unlock(&ctx->lock);
}

void recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info)
{
    struct jucx_context *ctx = (struct jucx_context *)request;
    ctx->length = info->length;
    ctx->sender_tag = info->sender_tag;
    jucx_request_callback(request, status);
}

void stream_recv_callback(void *request, ucs_status_t status, size_t length)
{
    struct jucx_context *ctx = (struct jucx_context *)request;
    ctx->length = length;
    jucx_request_callback(request, status);
}

UCS_PROFILE_FUNC(jobject, process_request, (request, callback), void *request, jobject callback)
{
    JNIEnv *env = get_jni_env();
    jobject jucx_request;

    if (UCS_PTR_IS_PTR(request)) {
        jucx_request = env->NewObject(jucx_request_cls, jucx_request_constructor,
                                      (native_ptr)request);
        struct jucx_context *ctx = (struct jucx_context *)request;
        ucs_recursive_spin_lock(&ctx->lock);
        if (ctx->status == UCS_INPROGRESS) {
            // request not completed yet, install user callback
            if (callback != NULL) {
                ctx->callback = env->NewGlobalRef(callback);
            }
            ctx->jucx_request = env->NewGlobalRef(jucx_request);
        } else {
            // request was completed whether by progress in other thread or inside
            // ucp_tag_recv_nb function call.
            set_jucx_request_completed(env, jucx_request, ctx, ctx->status);
            if (callback != NULL) {
                jucx_call_callback(callback, jucx_request, ctx->status);
            }
            jucx_context_reset(ctx);
            ucp_request_free(request);
        }
        ucs_recursive_spin_unlock(&ctx->lock);
    } else {
        jmethodID empty_constructor = env->GetMethodID(jucx_request_cls, "<init>", "()V");
        jucx_request = env->NewObject(jucx_request_cls, empty_constructor);
        set_jucx_request_completed(env, jucx_request, NULL, UCS_PTR_RAW_STATUS(request));
        if (UCS_PTR_IS_ERR(request)) {
            JNU_ThrowExceptionByStatus(env, UCS_PTR_STATUS(request));
            if (callback != NULL) {
                call_on_error(callback, UCS_PTR_STATUS(request));
            }
        } else if (callback != NULL) {
            call_on_success(callback, jucx_request);
        }
    }
    return jucx_request;
}

jobject process_completed_stream_recv(size_t length, jobject callback)
{
    JNIEnv *env = get_jni_env();
    jobject jucx_request = env->NewObject(jucx_request_cls, jucx_request_constructor, NULL);
    env->SetObjectField(jucx_request, native_id_field, NULL);
    env->SetLongField(jucx_request, recv_size_field, length);
    jucx_request_update_status(env, jucx_request, UCS_OK);
    if (callback != NULL) {
        jucx_call_callback(callback, jucx_request, UCS_OK);
    }
    return jucx_request;
}

void jucx_connection_handler(ucp_conn_request_h conn_request, void *arg)
{
    jobject jucx_conn_handler = reinterpret_cast<jobject>(arg);

    JNIEnv *env = get_jni_env();

    // Construct connection request class instance
    jclass conn_request_cls = env->FindClass("org/openucx/jucx/ucp/UcpConnectionRequest");
    jmethodID conn_request_constructor = env->GetMethodID(conn_request_cls, "<init>", "(J)V");
    jobject jucx_conn_request = env->NewObject(conn_request_cls, conn_request_constructor,
                                               (native_ptr)conn_request);

    // Call onConnectionRequest method
    jclass jucx_conn_hndl_cls = env->FindClass("org/openucx/jucx/ucp/UcpListenerConnectionHandler");
    jmethodID on_conn_request = env->GetMethodID(jucx_conn_hndl_cls, "onConnectionRequest",
                                       "(Lorg/openucx/jucx/ucp/UcpConnectionRequest;)V");
    env->CallVoidMethod(jucx_conn_handler, on_conn_request, jucx_conn_request);
    env->DeleteGlobalRef(jucx_conn_handler);
}


jobject new_rkey_instance(JNIEnv *env, ucp_rkey_h rkey)
{
    return env->NewObject(ucp_rkey_cls, ucp_rkey_cls_constructor, (native_ptr)rkey);
}

jobject new_tag_msg_instance(JNIEnv *env, ucp_tag_message_h msg_tag,
                             ucp_tag_recv_info_t *info_tag)
{
    return env->NewObject(ucp_tag_msg_cls, ucp_tag_msg_cls_constructor,
                         (native_ptr)msg_tag, info_tag->length, info_tag->sender_tag);
}
