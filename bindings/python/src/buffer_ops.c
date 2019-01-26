/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "buffer_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct data_buf *populate_buffer_region(void *src)
{
    struct data_buf *db = NULL;
    db = (struct data_buf *) malloc(sizeof(struct data_buf));
    db->buf = src;
    DEBUG_PRINT("allocated %p\n", db->buf);
    return db;
}

void *return_ptr_from_buf(struct data_buf *db)
{
    return (void *) db->buf;
}

struct data_buf *allocate_host_buffer(int length)
{
    struct data_buf *db = NULL;
    db = (struct data_buf *) malloc(sizeof(struct data_buf));
    db->buf = (void *) malloc(length);
    DEBUG_PRINT("allocated %p\n", db->buf);
    return db;
}

struct data_buf *allocate_cuda_buffer(int length)
{
    struct data_buf *db = NULL;
    db = (struct data_buf *) malloc(sizeof(struct data_buf));
    cudaMalloc((void **) &(db->buf), (size_t)length);
    DEBUG_PRINT("allocated %p\n", db->buf);
    return db;
}

int set_device(int device)
{
    CUDA_CHECK(cudaSetDevice(device));
    return 0;
}

int set_host_buffer(struct data_buf *db, int c, int length)
{
    memset((void *)db->buf, c, (size_t) length);
    return 0;
}

int set_cuda_buffer(struct data_buf *db, int c, int length)
{
    cudaMemset((void *)db->buf, c, (size_t) length);
    return 0;
}

int check_host_buffer(struct data_buf *db, int c, int length)
{
    char *tmp;
    int i;
    int errs = 0;

    tmp = (char *)db->buf;

    for (i = 0; i < length; i++) {
        if (c != (int) tmp[i]) errs++;
    }

    return errs;
}

int check_cuda_buffer(struct data_buf *db, int c, int length)
{
    char *tmp;
    int i;
    int errs = 0;

    tmp = (char *) malloc(sizeof(char) * length);
    cudaMemcpy((void *) tmp, (void *) db->buf, length, cudaMemcpyDeviceToHost);

    for (i = 0; i < length; i++) {
        if (c != (int) tmp[i]) errs++;
    }

    return errs;
}

int free_host_buffer(struct data_buf *db)
{
    free(db->buf);
    free(db);
    return 0;
}

int free_cuda_buffer(struct data_buf *db)
{
    cudaFree(db->buf);
    free(db);
    return 0;
}
