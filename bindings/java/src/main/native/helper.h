/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef HELPER_H_
#define HELPER_H_

#include <ucs/type/status.h>
#include <jni.h>
#include <cstdint>

typedef uintptr_t native_ptr;

static void log_error(const char* error);

JNIEXPORT void JNICALL JNU_ThrowException(JNIEnv *, const char *);

void JNU_ThrowExceptionByStatus(JNIEnv *, ucs_status_t);

/**
 * Bridge to call java callback from ucx callbacks (like request_init)
 */
struct callback_wrapper
{
    jobject java_obj;
    jmethodID java_callback;
    long request_size;
    JNIEnv* env;
    void _callback(void* request){
        jobject jbyte_buf = env->NewDirectByteBuffer(request, request_size);
        env->CallVoidMethod(java_obj, java_callback, jbyte_buf);
    }
    static void callback(void* request) { } //TODO: encapsulate into static method needed to call java callback.
};

#endif