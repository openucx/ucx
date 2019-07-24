/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "org_openucx_jucx_ucs_UcsConstants.h"
#include "jucx_common_def.h"

#include <ucs/type/thread_mode.h>

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucs_UcsConstants_loadConstants(JNIEnv *env, jclass cls)
{
    jclass thread_mode = env->FindClass("org/openucx/jucx/ucs/UcsConstants$ThreadMode");
    jfieldID field = env->GetStaticFieldID(thread_mode, "UCS_THREAD_MODE_MULTI", "I");
    env->SetStaticIntField(thread_mode, field, UCS_THREAD_MODE_MULTI);
}
