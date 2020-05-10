/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "org_openucx_jucx_ucp_UcpRequest.h"

#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

JNIEXPORT jboolean JNICALL
Java_org_openucx_jucx_ucp_UcpRequest_isCompletedNative(JNIEnv *env, jclass cls,
                                                       jlong ucp_req_ptr)
{
    return ucp_request_check_status((void *)ucp_req_ptr) != UCS_INPROGRESS;
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_ucp_UcpRequest_closeRequestNative(JNIEnv *env, jclass cls,
                                                        jlong ucp_req_ptr)
{
    ucp_request_free((void *)ucp_req_ptr);
}
