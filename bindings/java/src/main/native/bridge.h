/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef BRIDGE__H___
#define BRIDGE__H___

#include <cstdint>
#include <jni.h>

typedef uintptr_t native_ptr;

extern "C" {

/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    createWorkerNative
 * Signature: (ILorg/ucx/jucx/Worker/CompletionQueue;Lorg/ucx/jucx/Worker;)J
 */
JNIEXPORT jlong JNICALL Java_org_ucx_jucx_Bridge_createWorkerNative
  (JNIEnv *, jclass, jint, jobject, jobject);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    releaseWorkerNative
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_ucx_jucx_Bridge_releaseWorkerNative
  (JNIEnv *, jclass, jlong);
}

#endif
