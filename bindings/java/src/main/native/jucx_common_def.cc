/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "jucx_common_def.h"
extern "C" {
  #include <ucs/debug/debug.h>
}


extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void* reserved) {
   ucs_debug_disable_signals();
   return JNI_VERSION_1_1;
}

static inline void log_error(const char* error)
{
    ucs_error("JUCX - %s: %s \n", __FILE__, error);
}

/**
 * Throw a Java exception by name. Similar to SignalError.
 */
JNIEXPORT void JNICALL JNU_ThrowException(JNIEnv *env, const char *msg)
{
  jclass cls = env->FindClass("org/ucx/jucx/UcxException");
  log_error(msg);
  if (cls != 0) {/* Otherwise an exception has already been thrown */
    env->ThrowNew(cls, msg);
  }
}

void JNU_ThrowExceptionByStatus(JNIEnv *env, ucs_status_t status)
{
    JNU_ThrowException(env, ucs_status_string(status));
}
