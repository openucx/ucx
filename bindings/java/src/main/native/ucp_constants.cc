/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "org_ucx_jucx_ucp_UcpConstants.h"

#include <ucp/api/ucp.h>


#define JUCX_DEFINE_CONSTANT(_name) do { \
    jfieldID field = env->GetStaticFieldID(cls, #_name, "J"); \
    env->SetStaticLongField(cls, field, _name); \
} while(0)

/**
 * @brief Routine to set UCX constants in java
 *
 */
JNIEXPORT void JNICALL
Java_org_ucx_jucx_ucp_UcpConstants_loadConstants(JNIEnv *env, jclass cls)
{
    // UCP context parameters
    JUCX_DEFINE_CONSTANT(UCP_PARAM_FIELD_FEATURES);
    JUCX_DEFINE_CONSTANT(UCP_PARAM_FIELD_FEATURES);
    JUCX_DEFINE_CONSTANT(UCP_PARAM_FIELD_TAG_SENDER_MASK);
    JUCX_DEFINE_CONSTANT(UCP_PARAM_FIELD_MT_WORKERS_SHARED);
    JUCX_DEFINE_CONSTANT(UCP_PARAM_FIELD_ESTIMATED_NUM_EPS);

    // UCP configuration features
    JUCX_DEFINE_CONSTANT(UCP_FEATURE_TAG);
    JUCX_DEFINE_CONSTANT(UCP_FEATURE_RMA);
    JUCX_DEFINE_CONSTANT(UCP_FEATURE_AMO32);
    JUCX_DEFINE_CONSTANT(UCP_FEATURE_AMO64);
    JUCX_DEFINE_CONSTANT(UCP_FEATURE_WAKEUP);
    JUCX_DEFINE_CONSTANT(UCP_FEATURE_STREAM);
}
