/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "org_ucx_jucx_ucs_UcsConstants.h"
#include "jucx_common_def.h"

#include <ucs/type/thread_mode.h>

JNIEXPORT void JNICALL
Java_org_ucx_jucx_ucs_UcsConstants_loadConstants(JNIEnv *env, jclass cls)
{
    JUCX_DEFINE_ENUM(UCS_THREAD_MODE_SINGLE);
    JUCX_DEFINE_ENUM(UCS_THREAD_MODE_MULTI);
}
