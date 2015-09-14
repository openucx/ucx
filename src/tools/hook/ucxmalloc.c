/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/sys/compiler.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>

typedef struct {
  uint32_t canary;
  size_t data_length;
  size_t alloc_length;
} malloc_data_t;

#define CANARY_VALUE UINT32_MAX

static size_t compute_alloc_length(size_t length){
  size_t page_size = sysconf(_SC_PAGESIZE);
  if(length == 0){
    return page_size;
  }
  return (((length-1)/page_size)+1)*page_size;
}

static malloc_data_t *find_info(void *data){
  return data - sizeof(malloc_data_t);
}

static void check_for_freeable_pages(malloc_data_t *data){
  return;
}

int mallopt(int param, int value){
  return 0;
}

int malloc_trim(size_t pad){
  return 0;
}

void mtrace(void){
  return;
}

void muntrace(void){
  return;
}

struct mallinfo info;

struct mallinfo mallinfo(void){
  return info;
}

void malloc_stats(void){
  return;
}

void *malloc(size_t size){
  if(size == 0){
    return NULL;
  }
  void *new_region = mmap(NULL, size + sizeof(malloc_data_t), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_SHARED, 0, 0);
  if(new_region == MAP_FAILED){
    errno = ENOMEM;
    return NULL;
  }
  malloc_data_t *malloc_data = (malloc_data_t *)new_region;
  malloc_data->canary = CANARY_VALUE;
  malloc_data->data_length = size + sizeof(malloc_data_t);
  malloc_data->alloc_length = compute_alloc_length(malloc_data->data_length);
  new_region += sizeof(malloc_data_t);
  return new_region;
}

void *realloc(void *ptr, size_t size){
  if(ptr == NULL){
    return malloc(size);
  }else if(size == 0){
    void *base = malloc(1);
    malloc_data_t *new_data = find_info(base);
    new_data->data_length = 0;
    free(ptr);
    return base;
  }

  malloc_data_t *old_data = find_info(ptr);

  if(old_data->data_length < size){
    old_data->data_length = size;
    check_for_freeable_pages(old_data);
    return ptr;

  } else if(old_data->data_length == size) {
    return ptr;

  } else {

      if(old_data->alloc_length < size) {
        void *new_alloc = malloc(size);
        if(new_alloc == NULL){
          /* errno is already set */
          return NULL;
        }
        memcpy(new_alloc, ptr, old_data->data_length);
        free(ptr);
        return new_alloc;
      } else {
        old_data->data_length = size;
        return ptr;
      }
    }

}

void *calloc(size_t count, size_t size){
  return malloc(count * size);
}

void free(void *free_ptr){
  if(free_ptr == NULL){
    return;
  }

  malloc_data_t *old_data = find_info(free_ptr);
  if(old_data->canary != CANARY_VALUE){
    goto fail;
  }

  int ret_val = munmap((void *)old_data, old_data->alloc_length);

  if(ret_val == 0){
    return;
  }

fail:
    fprintf(stderr, "Got a bad value to free. Region is either corrupt or not allocated with malloc\n");
    abort();
}

static void UCS_F_CTOR ucxmalloc_init()
{
}

static void UCS_F_DTOR ucxmalloc_fini()
{
}
