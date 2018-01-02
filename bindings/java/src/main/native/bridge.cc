/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "bridge.h"
#include "worker.h"

#include <ucp/api/ucp.h>

#include <cstring>
#include <iostream>


#define ERR_EXIT(_msg, _ret)  do {                   \
                                print_error(_msg);   \
                                return _ret;         \
                              } while(0)

#define ERR_JUMP(_msg, _label)  do {                    \
                                    print_error(_msg);  \
                                    goto _label;        \
                                } while(0)

static JavaVM   *cached_jvm             = NULL;
static jfieldID field_comp_queue        = NULL;
static jfieldID field_worker_addr_arr   = NULL;

static context cached_ctx;


static void print_error(const char* error);


extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void* reserved) {
    cached_jvm = jvm;
    JNIEnv *env;

    if (jvm->GetEnv((void **) &env, JNI_VERSION_1_8)) {
        ERR_EXIT("JNI version 1.8 or higher required", JNI_ERR);
    }

    jclass queue_cls_data = env->FindClass("org/ucx/jucx/Worker$CompletionQueue");
    if (queue_cls_data == NULL) {
        ERR_EXIT("java org/ucx/jucx/Worker$CompletionQueue class was NOT found",
                 JNI_ERR);
    }

    field_comp_queue = env->GetFieldID(queue_cls_data, "completionBuff",
                                       "Ljava/nio/ByteBuffer;");
    if (field_comp_queue == NULL) {
        ERR_EXIT("could not get completionBuff's field id", JNI_ERR);
    }

    jclass worker_cls_data = env->FindClass("org/ucx/jucx/Worker");
    if (worker_cls_data == NULL) {
        ERR_EXIT("java org/ucx/jucx/Worker class was NOT found", JNI_ERR);
    }

    field_worker_addr_arr = env->GetFieldID(worker_cls_data,
                                            "workerAddress", "[B");
    if (field_worker_addr_arr == NULL) {
        ERR_EXIT("could not get workerAddress' field id", JNI_ERR);
    }

    return JNI_VERSION_1_8;
}

static void print_error(const char* error_msg) {
    fprintf(stderr, "[ERROR] JUCX - %s: %s\n", __FILE__, error_msg);
}

JNIEXPORT jlong JNICALL
Java_org_ucx_jucx_Bridge_createWorkerNative(JNIEnv *env, jclass cls,
                                            jint max_comp, jobject comp_queue,
                                            jobject jworker) {
    ucp_worker_params_t worker_params = { 0 };
    worker* worker_ptr = NULL;
    ucs_status_t status;
    jobject jbyte_buff;
    uint32_t cap = (uint32_t) max_comp;
    ucp_address_t* local_addr;
    size_t local_addr_len;
    jbyteArray jaddr_arr;
    jlong* addr_ptr;
    jbyte* local_addr_wrap;

    worker_params.field_mask    = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode   = UCS_THREAD_MODE_SINGLE;

    try {
        worker_ptr = new worker(&cached_ctx, cap, worker_params);
    } catch (const std::bad_alloc& ex) {
        ERR_JUMP("Failed to initialize ucp native worker", err);
    }

    status = worker_ptr->extract_worker_address(&local_addr, local_addr_len);
    if (!local_addr) {
        ERR_JUMP("Failed to get ucp worker native address", err_worker);
    }

    local_addr_wrap = new jbyte[local_addr_len];
    if (!local_addr_wrap) {
        ERR_JUMP("Allocation failure", err_local_addr);
    }
    memcpy(local_addr_wrap, local_addr, local_addr_len);

    jaddr_arr = env->NewByteArray(local_addr_len);
    if (!jaddr_arr) {
        ERR_JUMP("Failed to create Java byte[] object", err_worker_addr);
    }

    env->SetByteArrayRegion(jaddr_arr, 0, local_addr_len, local_addr_wrap);
    delete[] local_addr_wrap;
    worker_ptr->release_worker_address(local_addr);

    // Set the Java workerAddress field
    env->SetObjectField(jworker, field_worker_addr_arr, jaddr_arr);

    jbyte_buff = env->NewDirectByteBuffer(worker_ptr->get_event_queue(), cap);
    if (!jbyte_buff) {
        env->ExceptionClear();
        ERR_JUMP("Failed to create Java ByteBuffer object", err_worker);
    }

    // Set the completion queue field
    env->SetObjectField(comp_queue, field_comp_queue, jbyte_buff);

    return (native_ptr) worker_ptr;

err_worker_addr:
    delete[] local_addr_wrap;
err_local_addr:
    worker_ptr->release_worker_address(local_addr);
err_worker:
    delete worker_ptr;
err:
    return 0;
}

JNIEXPORT void JNICALL
Java_org_ucx_jucx_Bridge_releaseWorkerNative(JNIEnv *env, jclass cls,
                                             jlong worker_id) {
    worker* worker_ptr = (worker*) worker_id;
    delete worker_ptr;
}
