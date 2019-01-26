/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "common.h"

int set_device(int device);
struct data_buf *populate_buffer_region(void *src);
void *return_ptr_from_buf(struct data_buf *db);
struct data_buf *allocate_host_buffer(int length);
struct data_buf *allocate_cuda_buffer(int length);
int set_host_buffer(struct data_buf *db, int c, int length);
int set_cuda_buffer(struct data_buf *db, int c, int length);
int check_host_buffer(struct data_buf *db, int c, int length);
int check_cuda_buffer(struct data_buf *db, int c, int length);
int free_host_buffer(struct data_buf *buf);
int free_cuda_buffer(struct data_buf *buf);
