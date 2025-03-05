/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <cuda.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <unistd.h>


#define MAX_THREADS 2
#define MAX_CTX     2
#define MAX_DEV     4


/* Context and associated resources */
typedef struct {
    CUcontext ctx;
    CUstream  stream;
    void      *mem;
    void      *mem_managed;
} context_t;

context_t context[MAX_CTX * MAX_THREADS * MAX_DEV];

typedef struct thread {
    int             tid;
    int             signal;
    struct thread   *base;
    pthread_t       thread;
    pthread_cond_t  cond;
    pthread_mutex_t mutex;
} thread_t;

static thread_t thread[MAX_THREADS];

static int device_count;

#define CHECK(status) ({ \
    cudaError_t _err = (status); \
    if (_err != cudaSuccess) { \
        printf("error: %s in at %s:%d\n", \
               cudaGetErrorString(_err), __FILE__,  __LINE__); \
        exit(EXIT_FAILURE); \
    } \
})

#define CHECK_D(status) ({ \
    CUresult _err = (status); \
    if (_err != CUDA_SUCCESS) { \
        printf("error: err=%d in at %s:%d\n", _err, __FILE__,  __LINE__); \
        exit(EXIT_FAILURE); \
    } \
})


/* Initialize all contexts and their streams */
static void setup_contexts(context_t *context, int tid, int count)
{
    int i, j, active, index;
    unsigned int flags;
    CUcontext ctx;
    CUdevice device;

    for (i = 0; i < count; i++) {
        index = ((i * MAX_THREADS) + tid) * MAX_CTX;

        CHECK(cuDeviceGet(&device, i));
        CHECK(cuCtxCreate(&context[index].ctx, 0, device));
        CHECK_D(cuStreamCreate(&context[index].stream, CU_STREAM_NON_BLOCKING));

        for (j = 1; j < MAX_CTX; j++) {
            CHECK(cuDeviceGet(&device, i));
            CHECK_D(cuCtxCreate(&context[index + j].ctx, 0, device));
            CHECK_D(cuStreamCreate(&context[index + j].stream, CU_STREAM_NON_BLOCKING));
        }
    }

    for (i = 0; i < count; i++) {
        CHECK_D(cuDevicePrimaryCtxGetState(i, &flags, &active));
        printf("#tid%d GPU%d primary_flags=%x primary_active=%d\n",
               tid, i, flags, active);

        index = ((i * MAX_THREADS) + tid) * MAX_CTX;

        for (j = 0; j < MAX_CTX; j++) {
            CHECK_D(cuStreamGetCtx(context[index + j].stream, &ctx));
            printf("  ctx%d=%p stream=%p %s%s\n", j,
                   context[index + j].ctx, context[index + j].stream,
                   context[index + j].ctx == ctx? "ok" : "KO",
                   j == 0? " (primary)" : "");
        }
    }
}

#define KB 1024LLU
#define MB (KB * 1024)

static size_t sizes[] = {
    32, 64, 65, 512*KB, 10*MB, 0
};

static void setup_malloc(context_t *context, int tid, int count,
                         size_t size)
{
    int i, j, index;

    for (i = 0; i < count; i++) {
        for (j = 0; j < MAX_CTX; j++) {
            index = ((i * MAX_THREADS) + tid) * MAX_CTX + j;

            if (context[index].mem_managed) {
                cudaFree(context[index].mem_managed);
            }
            if (context[index].mem) {
                cudaFree(context[index].mem);
            }

            CHECK(cudaSetDevice(i));
            CHECK(cudaMallocManaged(&context[index].mem_managed, size,
                                    cudaMemAttachGlobal));
            CHECK(cudaMalloc(&context[index].mem, size));
        }
    }
}

static int get_gpu(int index)
{
    return index / (MAX_CTX * MAX_THREADS);
}

static int get_ctx(int index)
{
    return index % MAX_CTX;
}

static void test_copy(context_t *context, int tid, int count,
                      size_t size)
{
    static int done = 0;
    int j, k, l;
    CUdeviceptr ptr_a, ptr_b, ptr_a_managed;
    CUstream stream;

    /* each source context (every thread every device) */
    for (j = 0; j < MAX_CTX * MAX_THREADS * count; j++) {
    /* each destination context (every thread every device) */
    for (k = 0; k < MAX_CTX * MAX_THREADS * count; k++) {
    /* each stream (every thread every device) */
    for (l = 0; l < MAX_CTX * MAX_THREADS * count; l++) {
        ptr_a         = (CUdeviceptr)context[j].mem;
        ptr_a_managed = (CUdeviceptr)context[j].mem_managed;
        ptr_b         = (CUdeviceptr)context[k].mem;
        stream        = context[l].stream;

        if (!tid && !done) {
            printf("size=%zu a=%llu/GPU%d/%d "
                   "b=%llu/GPU%d/%d a_managed=%llu stream=%p/GPU%d/%d\n",
                   size,
                   ptr_a, get_gpu(j), get_ctx(j),
                   ptr_b, get_gpu(k), get_ctx(k),
                   ptr_a_managed,
                   stream, get_gpu(l), get_ctx(l));
        }

        CHECK_D(cuMemcpyAsync(ptr_a, ptr_b, size, stream));
        CHECK_D(cuMemcpy(ptr_a, ptr_b, size));
        CHECK_D(cuMemcpyAsync(ptr_a_managed, ptr_b, size, stream));
        CHECK_D(cuMemcpy(ptr_a_managed, ptr_b, size));

        CHECK_D(cuStreamSynchronize(stream));

    }}}

    if (!tid) {
        done = 1;
    }
}

static void wait(thread_t *t)
{
    pthread_mutex_lock(&t->mutex);
    while (t->signal == 0) {
        pthread_cond_wait(&t->cond, &t->mutex);
    }
    t->signal = 0;
    pthread_mutex_unlock(&t->mutex);
}

static void signal(thread_t *t)
{
    pthread_mutex_lock(&t->mutex);
    t->signal = 1;
    pthread_cond_signal(&t->cond);
    pthread_mutex_unlock(&t->mutex);
}

static void signal_next(thread_t *t)
{
    signal(&t->base[(t->tid + 1) % MAX_THREADS]);
}

static void *worker_cb(void *arg)
{
    thread_t *t = (thread_t *)arg;
    size_t *s;

    wait(t);
    setup_contexts(context, t->tid, device_count);
    signal_next(t);

    for (s = sizes; *s; s++) {
        wait(t);
        setup_malloc(context, t->tid, device_count, *s);
        signal_next(t);

        if (t->tid == 0) {
            printf("test copy size=%zu\n", *s);
        }

        wait(t);
        test_copy(context, t->tid, device_count, *s);
        signal_next(t);
    }

    wait(t);
    signal_next(t);
    return NULL;
}

int main(int argc, char *argv[])
{
    int i, ret;

    CHECK_D(cuInit(0));
    CHECK_D(cuDeviceGetCount(&device_count));

    device_count = device_count > MAX_DEV? MAX_DEV : device_count;
    printf("Using %d CUDA device(s)\n", device_count);

    for (i = 0; i < MAX_THREADS; i++) {
        thread[i].tid  = i;
        thread[i].base = thread;

        pthread_mutex_init(&thread[i].mutex, NULL);
        pthread_cond_init(&thread[i].cond, NULL);

        if (!i) {
            continue;
        }

        ret = pthread_create(&thread[i].thread, NULL, worker_cb, &thread[i]);
        if (ret) {
            fprintf(stderr, "error: thread%d creation failed\n", i);
            return -1;
        }
    }

    signal(&thread[0]);
    worker_cb(&thread[0]);

    for (i = 1; i < MAX_THREADS; i++) {
        ret = pthread_join(thread[i].thread, NULL);
        if (ret) {
            fprintf(stderr, "error: thread%d failed to join\n", i);
            return -1;
        }
    }

    return 0;
}
