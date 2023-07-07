/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sys/mman.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>

#include <ucs/sys/preprocessor.h>
#include <ucm/api/ucm.h>
#include <ucm/util/sys.h>


#define DEFAULT_THREAD_COUNT   16
#define DEFAULT_THREAD_TIMEOUT 1


typedef struct {
    const char   *libname;
    int          thread_count;
    pthread_t    *threads;
    volatile int stop;
    double       thread_timeout;
    void         *dl_handle;
} context_t;

static ucs_status_t ctx_init(context_t *ctx, const char *filename,
                             int thread_count, int thread_timeout)
{
    pthread_t *threads;

    threads = calloc(thread_count, sizeof(*threads));
    if (threads == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ctx->dl_handle      = NULL;
    ctx->libname        = filename;
    ctx->thread_count   = thread_count;
    ctx->threads        = threads;
    ctx->stop           = 0;
    ctx->thread_timeout = thread_timeout;

    return UCS_OK;
}

static ucs_status_t ctx_load_library(context_t *ctx)
{
    printf("loading library to trigger hooks installation\n");

    dlerror();
    ctx->dl_handle = dlopen(ctx->libname, RTLD_NOW);
    if (ctx->dl_handle == NULL) {
        printf("dlopen(filename=\"%s\", RTLD_NOW) failed: %s\n", ctx->libname,
               dlerror());
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static void *thread_handler(void *arg)
{
    context_t *ctx  = arg;
    double deadline = ucm_get_time() + ctx->thread_timeout;

    /*
     * Many threads racing for read lock are preventing ucm to finish testing
     * the installed hooks, so we make sure to stop the threads after
     * reasonable time.
     */
    do {
        if (mmap(NULL, 0, PROT_READ | PROT_WRITE, MAP_PRIVATE, -1, 0) !=
            MAP_FAILED) {
            printf("failed to return mmap_call() failure\n");
            exit(EXIT_FAILURE);
        }
    } while (!ctx->stop &&
             ((ctx->thread_timeout == 0) || (ucm_get_time() < deadline)));

    return NULL;
}

static void ctx_stop_threads(context_t *ctx, int count)
{
    int i;

    printf("stopping %d threads\n", count);

    ctx->stop = 1;
    for (i = 0; i < count; i++) {
        pthread_join(ctx->threads[i], NULL);
    }
}

static ucs_status_t ctx_start_threads(context_t *ctx)
{
    int i, ret;

    for (i = 0; i < ctx->thread_count; i++) {
        ret = pthread_create(&ctx->threads[i], NULL, thread_handler, ctx);
        if (ret < 0) {
            printf("pthread_create(thread #%d) failed: %m\n", i);
            goto err;
        }
    }

    printf("started %d threads\n", ctx->thread_count);
    return UCS_OK;

err:
    ctx_stop_threads(ctx, i);
    return UCS_ERR_NO_RESOURCE;
}

static void ctx_free(context_t *ctx)
{
    if (ctx->dl_handle != NULL) {
        dlclose(ctx->dl_handle);
    }
    free(ctx->threads);
}

static void usage(const char *argv0)
{
    printf("Usage: %s [options]\n", argv0);
    printf("Options:\n");
    printf("  -n :  Number of threads racing with function patching (default: "
           "%d)\n",
           DEFAULT_THREAD_COUNT);
    printf("  -d :  Disable thread timeout (default %d sec)\n",
           DEFAULT_THREAD_TIMEOUT);
    printf("  -h :  Display help message\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    int thread_count   = DEFAULT_THREAD_COUNT;
    int thread_timeout = DEFAULT_THREAD_TIMEOUT;
    int delay_usec     = 200 * 1000;
    int c, ret;
    context_t ctx;
    ucs_status_t status;

    while ((c = getopt(argc, argv, "n:dh")) != -1) {
        switch (c) {
        case 'n':
            thread_count = atoi(optarg);
            break;
        case 'd':
            thread_timeout = 0;
            break;
        case 'h':
        default:
            usage(argv[0]);
            return -1;
        }
    }

    status = ctx_init(&ctx, UCS_PP_MAKE_STRING(LIB_PATH), thread_count,
                      thread_timeout);
    if (status != UCS_OK) {
        return -1;
    }

    printf("thread timeout: %d sec\n", thread_timeout);

    status = ctx_start_threads(&ctx);
    if (status != UCS_OK) {
        printf("failed to start threads\n");
        ret = -1;
        goto out;
    }

    usleep(delay_usec); /* make sure threads are running */

    /* trigger ucm memory patching */
    status = ctx_load_library(&ctx);
    if (status != UCS_OK) {
        printf("failed to load library\n");
        ret = -1;
        goto out_stop_threads;
    }

    usleep(delay_usec); /* make sure threads have time to race */

    ret = 0;

out_stop_threads:
    ctx_stop_threads(&ctx, ctx.thread_count);
out:
    ctx_free(&ctx);
    return ret;
}
