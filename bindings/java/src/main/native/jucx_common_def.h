/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef HELPER_H_
#define HELPER_H_

#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>

#include <jni.h>

#include <cstdint>


typedef uintptr_t native_ptr;

static void log_error(const char* error);

JNIEXPORT void JNICALL JNU_ThrowException(JNIEnv *, const char *);

void JNU_ThrowExceptionByStatus(JNIEnv *, ucs_status_t);

#endif
