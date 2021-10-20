/**
 * Copyright (C) Mellanox Technologies Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_H_
#define UCP_DT_H_

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_types.h>
#include <ucs/arch/cpu.h>
#include <ucs/profile/profile.h>
#include <uct/api/uct.h>


/**
 * Datatype classification
 */
typedef enum ucp_dt_type          ucp_dt_class_t;


/**
 * Memory registration state of a buffer/operation
 */
typedef struct ucp_dt_reg {
    ucp_md_map_t                  md_map;    /* Map of used memory domains */
    uct_mem_h                     memh[UCP_MAX_OP_MDS];
} ucp_dt_reg_t;


/**
 * State of progressing sent/receive operation on a datatype.
 */
typedef struct ucp_dt_state {
    size_t                        offset;  /* Total offset in overall payload. */
    union {
        ucp_dt_reg_t              contig;
        struct {
            size_t                iov_offset;     /* Offset in the IOV item */
            size_t                iovcnt_offset;  /* The IOV item to start copy */
            size_t                iovcnt;         /* Number of IOV buffers */
            ucp_dt_reg_t          *dt_reg;        /* Pointer to IOV memh[iovcnt] */
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


/**
 * This type describes a datatype packing function, used to pack into
 * a contiguous buffer.
 * 
 * @param [in]  worker   UCP worker
 * @param [out] dest     Pack into this buffer
 * @param [in]  src      Source data to pack
 * @param [in]  length   Length of the data to pack
 * @param [in]  mem_type Memory type of the source data
 * 
 * @return Error code as defined by @ref ucs_status_t
 */
typedef ucs_status_t (*ucp_dt_pack_func_t)(ucp_worker_h worker, void *dest,
                                           const void *src, size_t length,
                                           ucs_memory_type_t mem_type);


/**
 * This type describes a datatype unpacking function, used to unpack from
 * a contiguous buffer.
 * 
 * @param [in]  worker   UCP worker
 * @param [out] dest     Unpack into this buffer
 * @param [in]  src      Source buffer to unpack
 * @param [in]  length   Length of the data to unpack
 * @param [in]  mem_type Memory type of the dest data
 * 
 * @return Error code as defined by @ref ucs_status_t
 */
typedef ucs_status_t (*ucp_dt_unpack_func_t)(ucp_worker_h worker, void *dest,
                                             const void *src, size_t length,
                                             ucs_memory_type_t mem_type);


extern const char *ucp_datatype_class_names[];

size_t ucp_dt_pack(ucp_worker_h worker, ucp_datatype_t datatype,
                   ucs_memory_type_t mem_type, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length);

ucs_status_t ucp_mem_type_pack(ucp_worker_h worker, void *dest,
                               const void *src, size_t length,
                               ucs_memory_type_t mem_type);

ucs_status_t ucp_mem_type_unpack(ucp_worker_h worker, void *buffer,
                                 const void *recv_data, size_t recv_length,
                                 ucs_memory_type_t mem_type);


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_memcpy_pack_unpack(ucp_worker_h worker, void *buffer, const void *data,
                       size_t length, ucs_memory_type_t mem_type)
{
    UCS_PROFILE_CALL(ucs_memcpy_relaxed, buffer, data, length);
    return UCS_OK;
}


#endif /* UCP_DT_H_ */
