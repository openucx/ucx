/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef BRIDGE__H___
#define BRIDGE__H___

#include <cstdint>
#include <jni.h>

extern "C" {

/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    createWorkerNative
 * Signature: (ILorg/ucx/jucx/Worker/CompletionQueue;Lorg/ucx/jucx/Worker;Z)J
 */
JNIEXPORT jlong JNICALL Java_org_ucx_jucx_Bridge_createWorkerNative
  (JNIEnv *, jclass, jint, jobject, jobject, jboolean);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    destroyWorkerNative
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_ucx_jucx_Bridge_destroyWorkerNative
  (JNIEnv *, jclass, jlong);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    createEndPointNative
 * Signature: (J[B)J
 */
JNIEXPORT jlong JNICALL Java_org_ucx_jucx_Bridge_createEndPointNative
  (JNIEnv *, jclass, jlong, jbyteArray);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    destroyEndPointNative
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_ucx_jucx_Bridge_destroyEndPointNative
  (JNIEnv *, jclass, jlong);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    streamSendNative
 * Signature: (JJJIJ)Z
 */
JNIEXPORT jboolean JNICALL Java_org_ucx_jucx_Bridge_streamSendNative
  (JNIEnv *, jclass, jlong, jlong, jlong, jint, jlong);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    streamRecvNative
 * Signature: (JJJIJ)Z
 */
JNIEXPORT jboolean JNICALL Java_org_ucx_jucx_Bridge_streamRecvNative
  (JNIEnv *, jclass, jlong, jlong, jlong, jint, jlong);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    progressWorkerNative
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_ucx_jucx_Bridge_progressWorkerNative
  (JNIEnv *, jclass, jlong);


/*
 * Class:     org_ucx_jucx_Bridge
 * Method:    isMultiThreadSupportEnabledNative
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_org_ucx_jucx_Bridge_isMultiThreadSupportEnabledNative
  (JNIEnv *, jclass);
}

#endif
