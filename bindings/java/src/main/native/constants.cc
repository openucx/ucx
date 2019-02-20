/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "org_ucx_jucx_UcxConstants.h"

#include <ucp/api/ucp.h>

/**
 * @brief Routine to set UCX constants in java
 *
 */
JNIEXPORT void JNICALL
Java_org_ucx_jucx_UcxConstants_loadConstants(JNIEnv *env, jclass cls)
{
    jfieldID field;

    // UCP context parameters
    field = env->GetStaticFieldID(cls, "UCP_PARAM_FIELD_FEATURES", "J");
    env->SetStaticLongField(cls, field, UCP_PARAM_FIELD_FEATURES);

    field = env->GetStaticFieldID(cls, "UCP_PARAM_FIELD_TAG_SENDER_MASK", "J");
    env->SetStaticLongField(cls, field, UCP_PARAM_FIELD_TAG_SENDER_MASK);

    field = env->GetStaticFieldID(cls, "UCP_PARAM_FIELD_MT_WORKERS_SHARED", "J");
    env->SetStaticLongField(cls, field, UCP_PARAM_FIELD_MT_WORKERS_SHARED);

    field = env->GetStaticFieldID(cls, "UCP_PARAM_FIELD_ESTIMATED_NUM_EPS", "J");
    env->SetStaticLongField(cls, field, UCP_PARAM_FIELD_ESTIMATED_NUM_EPS);

    // UCP configuration features
    field = env->GetStaticFieldID(cls, "UCP_FEATURE_TAG", "J");
    env->SetStaticLongField(cls, field, UCP_FEATURE_TAG);

    field = env->GetStaticFieldID(cls, "UCP_FEATURE_RMA", "J");
    env->SetStaticLongField(cls, field, UCP_FEATURE_RMA);

    field = env->GetStaticFieldID(cls, "UCP_FEATURE_AMO32", "J");
    env->SetStaticLongField(cls, field, UCP_FEATURE_AMO32);

    field = env->GetStaticFieldID(cls, "UCP_FEATURE_AMO64", "J");
    env->SetStaticLongField(cls, field, UCP_FEATURE_AMO64);

    field = env->GetStaticFieldID(cls, "UCP_FEATURE_WAKEUP", "J");
    env->SetStaticLongField(cls, field, UCP_FEATURE_WAKEUP);

    field = env->GetStaticFieldID(cls, "UCP_FEATURE_STREAM", "J");
    env->SetStaticLongField(cls, field, UCP_FEATURE_STREAM);
}
