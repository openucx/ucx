/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ROCM_IPC_MD_H
#define ROCM_IPC_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/sys.h>
#include <hsa_ext_amd.h>
#include <pthread.h>
#include <string.h>


typedef struct uct_rocm_ipc_cache uct_rocm_ipc_cache_t;

typedef struct uct_rocm_ipc_component {
    uct_component_t super;
} uct_rocm_ipc_component_t;

extern uct_rocm_ipc_component_t uct_rocm_ipc_component;

typedef struct uct_rocm_ipc_md {
    struct uct_md super;
} uct_rocm_ipc_md_t;

typedef struct uct_rocm_ipc_md_config {
    uct_md_config_t super;
} uct_rocm_ipc_md_config_t;

/* Memory mapped with hipMemMap is shared by exporting its allocation as a
 * dma-buf file descriptor, which needs the HSA virtual memory API */
#define UCT_ROCM_IPC_HAVE_VMEM_IPC \
    (HAVE_DECL_HSA_EXT_POINTER_TYPE_HSA_VMEM && \
     HAVE_HSA_AMD_VMEM_RETAIN_ALLOC_HANDLE && \
     HAVE_HSA_AMD_VMEM_EXPORT_SHAREABLE_HANDLE && \
     HAVE_HSA_AMD_VMEM_IMPORT_SHAREABLE_HANDLE && HAVE_HSA_AMD_VMEM_MAP && \
     HAVE_HSA_AMD_VMEM_UNMAP)

typedef enum {
    /* The memory cannot be shared, so peers must reach it another way */
    UCT_ROCM_IPC_KEY_HANDLE_TYPE_NONE,
    /* hsa_amd_ipc_memory_create handle */
    UCT_ROCM_IPC_KEY_HANDLE_TYPE_IPC,
    /* dma-buf fd of a VMM allocation, served by the exporter's fd server */
    UCT_ROCM_IPC_KEY_HANDLE_TYPE_VMEM_FD
} uct_rocm_ipc_key_handle_type_t;

typedef struct uct_rocm_ipc_key {
    hsa_amd_ipc_memory_t ipc;
    uintptr_t            address; /* Base address of the shared allocation */
    size_t               length;  /* Length of the shared allocation */
    int                  dev_num;
    pid_t                pid; /* PID of the process that owns the memory */
    ucs_sys_ns_t         pid_ns; /* PID namespace of the owner process */
    uint8_t              handle_type; /* uct_rocm_ipc_key_handle_type_t */
    uint64_t             server_id; /* VMEM_FD: fd server of the owner */
    uint64_t             fd_id; /* VMEM_FD: exported allocation */
} uct_rocm_ipc_key_t;


/* Whether two keys refer to the same shared allocation */
static inline int uct_rocm_ipc_key_is_same(const uct_rocm_ipc_key_t *key1,
                                           const uct_rocm_ipc_key_t *key2)
{
    if (key1->handle_type != key2->handle_type) {
        return 0;
    }

    if (key1->handle_type == UCT_ROCM_IPC_KEY_HANDLE_TYPE_VMEM_FD) {
        return (key1->server_id == key2->server_id) &&
               (key1->fd_id == key2->fd_id);
    }

    return memcmp(&key1->ipc, &key2->ipc, sizeof(key1->ipc)) == 0;
}

#endif
