/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef HELPER_H_
#define HELPER_H_

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/profile/profile.h>
#include <ucs/type/spinlock.h>

#include <jni.h>


typedef uintptr_t native_ptr;

#define JUCX_DEFINE_LONG_CONSTANT(_name) do { \
    jfieldID field = env->GetStaticFieldID(cls, #_name, "J"); \
    if (field != NULL) { \
        env->SetStaticLongField(cls, field, _name); \
    } \
} while(0)

#define JUCX_DEFINE_INT_CONSTANT(_name) do { \
    jfieldID field = env->GetStaticFieldID(cls, #_name, "I"); \
    if (field != NULL) { \
        env->SetStaticIntField(cls, field, _name); \
    } \
} while(0)

/**
 * Throw a Java exception by name. Similar to SignalError.
 */
#define JNU_ThrowException(_env, _msg) do { \
    jclass _cls = _env->FindClass("org/openucx/jucx/UcxException"); \
    ucs_error("JUCX: %s", _msg); \
    if (_cls != 0) { /* Otherwise an exception has already been thrown */ \
        _env->ThrowNew(_cls, _msg); \
    } \
} while(0)

#define JNU_ThrowExceptionByStatus(_env, _status) do { \
    jclass _cls = _env->FindClass("org/openucx/jucx/UcxException"); \
    jmethodID _constr = _env->GetMethodID(_cls, "<init>", "(Ljava/lang/String;I)V"); \
    jstring _error_msg = _env->NewStringUTF(ucs_status_string(_status)); \
    jthrowable _ex = \
    static_cast<jthrowable>(_env->NewObject(_cls, _constr, _error_msg, _status)); \
    _env->Throw(_ex); \
} while(0)

/**
 * @brief Utility to convert Java InetSocketAddress class (corresponds to the Network Layer 4
 * and consists of an IP address and a port number) to corresponding sockaddr_storage struct.
 * Supports IPv4 and IPv6.
 */
bool j2cInetSockAddr(JNIEnv *env, jobject sock_addr, sockaddr_storage& ss, socklen_t& sa_len);

/**
 * @brief Utility to convert c sockaddr to java InetSocketAddress
 */
jobject c2jInetSockAddr(JNIEnv *env, const sockaddr_storage* ss);

/**
 * @brief Get the jni env object. To be able to call java methods from ucx async callbacks.
 */
JNIEnv* get_jni_env();

/**
 * @brief Send callback used to invoke java callback class on completion of ucp operations.
 */
void jucx_request_callback(void *request, ucs_status_t status, void *user_data);

/**
 * @brief Recv callback used to invoke java callback class on completion of ucp tag_recv_nb operation.
 */
void recv_callback(void *request, ucs_status_t status, const ucp_tag_recv_info_t *info,
                   void *user_data);

/**
 * @brief Recv callback used to invoke java callback class on completion of ucp stream_recv_nb operation.
 */
void stream_recv_callback(void *request, ucs_status_t status, size_t length, void *user_data);

/**
 * @brief Active message receive callback.
 */
ucs_status_t am_recv_callback(void *arg, const void *header, size_t header_length, void *data, size_t length,
                              const ucp_am_recv_param_t *param);

/**
 * @ingroup JUCX_REQ
 * @brief Utility to allocate jucx request and set appropriate java callback in it.
 */
jobject jucx_request_allocate(JNIEnv *env, jobject callback, ucp_request_param_t *param,
                              jobject request_params);

/**
 * @ingroup JUCX_REQ
 * @brief Utility to set iov verctor in jucx_request, to release it on completion.
 */
void jucx_request_set_iov(JNIEnv *env, jobject request, ucp_dt_iov_t* iovec);

/**
 * @ingroup JUCX_REQ
 * @brief Utility to update status of JUCX request to corresponding ucx request.
 */
void jucx_request_update_status(JNIEnv *env, jobject jucx_request, ucs_status_t status);

/**
 * @ingroup JUCX_REQ
 * @brief Utility to set recv length in JUCX request.
 */
void jucx_request_update_recv_length(JNIEnv *env, jobject jucx_request, size_t rlength);

/**
 * @ingroup JUCX_REQ
 * @brief Utility to set sender tag in JUCX request.
 */
void jucx_request_update_sender_tag(JNIEnv *env, jobject jucx_request, ucp_tag_t sender_tag);

/**
 * @brief Function to handle result of ucx function submition, to handle immidiate completion.
 */
void process_request(JNIEnv *env, jobject request, ucs_status_ptr_t status);

void jucx_connection_handler(ucp_conn_request_h conn_request, void *arg);

/**
 * @brief Creates new jucx rkey class.
 */
jobject new_rkey_instance(JNIEnv *env, ucp_rkey_h rkey);

/**
 * @brief Creates new jucx tag_msg class.
 */
jobject new_tag_msg_instance(JNIEnv *env, ucp_tag_message_h msg_tag,
                             ucp_tag_recv_info_t *info_tag);

/**
 * @brief Creates iov vector from array of addresses and sizes
 */
UCS_F_ALWAYS_INLINE ucp_dt_iov_t* get_ucp_iov(JNIEnv *env,
                                              jlongArray addr_array, jlongArray size_array,
                                              int &iovcnt)
{
    iovcnt = env->GetArrayLength(addr_array);
    ucp_dt_iov_t* iovec = (ucp_dt_iov_t*)ucs_malloc(sizeof(*iovec) * iovcnt, "JUCX iov vector");
    if (ucs_unlikely(iovec == NULL)) {
        ucs_error("failed to allocate buffer for %d iovec", iovcnt);
        JNU_ThrowException(env, "failed to allocate buffer for iovec");
        return NULL;
    }

    jlong* addresses = env->GetLongArrayElements(addr_array, NULL);
    jlong* sizes = env->GetLongArrayElements(size_array, NULL);

    for (int i = 0; i < iovcnt; i++) {
        iovec[i].buffer = (void *)addresses[i];
        iovec[i].length = sizes[i];
    }

    env->ReleaseLongArrayElements(addr_array, addresses, 0);
    env->ReleaseLongArrayElements(size_array, sizes, 0);

    return iovec;
}

#endif
