/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2016. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_H_
#define UCP_DT_H_

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_types.h>
#include <ucs/arch/cpu.h>
#include <ucs/profile/profile.h>
#include <uct/api/uct.h>


/**
 * Datatype classification
 */
typedef enum ucp_dt_type          ucp_dt_class_t;


/**
 * State of progressing sent/receive operation on a datatype.
 */
typedef struct ucp_dt_state {
    size_t                        offset;  /* Total offset in overall payload. */
    union {
        struct {
            ucp_mem_h             memh;      /* Pointer to memh */
        } contig;
        struct {
            size_t                iov_offset;     /* Offset in the IOV item */
            size_t                iovcnt_offset;  /* The IOV item to start copy */
            size_t                iovcnt;         /* Number of IOV buffers */
            ucp_mem_h             *memhs;         /* Pointer to IOV memh[iovcnt] */
        } iov;
        struct {
            void                  *state;
        } generic;
    } dt;
} ucp_dt_state_t;


/**
 * UCP layer memory information
 */
typedef struct {
    uint8_t          type;    /**< Memory type, use uint8 for compact size */
    ucs_sys_device_t sys_dev; /**< System device index */
} ucp_memory_info_t;


extern const char *ucp_datatype_class_names[];

size_t ucp_dt_pack(ucp_worker_h worker, ucp_datatype_t datatype,
                   ucs_memory_type_t mem_type, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length);

void ucp_mem_type_pack(ucp_worker_h worker, void *dest, const void *src,
                       size_t length, ucs_memory_type_t mem_type);

void ucp_mem_type_unpack(ucp_worker_h worker, void *buffer,
                         const void *recv_data, size_t recv_length,
                         ucs_memory_type_t mem_type);


static UCS_F_ALWAYS_INLINE void
ucp_memcpy_pack(void *buffer, const void *data, size_t length,
                size_t total_len, const char *name)
{
    UCS_PROFILE_NAMED_CALL(name, ucs_memcpy_relaxed, buffer, data, length,
                           UCS_ARCH_MEMCPY_NT_DEST, total_len);
}

static UCS_F_ALWAYS_INLINE void
ucp_memcpy_unpack(void *buffer, const void *data, size_t length,
                  size_t total_len, const char *name)
{
    UCS_PROFILE_NAMED_CALL(name, ucs_memcpy_relaxed, buffer, data, length,
                           UCS_ARCH_MEMCPY_NT_SOURCE, total_len);
}

#endif /* UCP_DT_H_ */
