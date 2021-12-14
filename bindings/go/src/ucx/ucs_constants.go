/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

//#include <ucp/api/ucp.h>
//static inline const char* ucxgo_get_ucs_mem_type_name(ucs_memory_type_t idx) {
//    return ucs_memory_type_names[idx];
//}
import "C"

type UcsThreadMode int

const (
	UCS_THREAD_MODE_SINGLE UcsThreadMode = C.UCS_THREAD_MODE_SINGLE
	UCS_THREAD_MODE_MULTI  UcsThreadMode = C.UCS_THREAD_MODE_MULTI
)

type UcsMemoryType int

const (
	UCS_MEMORY_TYPE_HOST         UcsMemoryType = C.UCS_MEMORY_TYPE_HOST         /**< Default system memory */
	UCS_MEMORY_TYPE_CUDA         UcsMemoryType = C.UCS_MEMORY_TYPE_CUDA         /**< NVIDIA CUDA memory */
	UCS_MEMORY_TYPE_CUDA_MANAGED UcsMemoryType = C.UCS_MEMORY_TYPE_CUDA_MANAGED /**< NVIDIA CUDA managed (or unified) memory */
	UCS_MEMORY_TYPE_ROCM         UcsMemoryType = C.UCS_MEMORY_TYPE_ROCM         /**< AMD ROCM memory */
	UCS_MEMORY_TYPE_ROCM_MANAGED UcsMemoryType = C.UCS_MEMORY_TYPE_ROCM_MANAGED /**< AMD ROCM managed system memory */
	UCS_MEMORY_TYPE_UNKNOWN      UcsMemoryType = C.UCS_MEMORY_TYPE_UNKNOWN
)

func (m UcsMemoryType) String() string {
	return C.GoString(C.ucxgo_get_ucs_mem_type_name(C.ucs_memory_type_t(m)))
}

// Checks whether context's memory type mask
// (received via UcpContext.MemoryTypesMask()) supports particular memory type.
func IsMemTypeSupported(memType UcsMemoryType, mask uint64) bool {
	return ((1 << memType) & mask) != 0
}

type UcsStatus int

const (
	UCS_OK                         UcsStatus = C.UCS_OK
	UCS_INPROGRESS                 UcsStatus = C.UCS_INPROGRESS
	UCS_ERR_NO_MESSAGE             UcsStatus = C.UCS_ERR_NO_MESSAGE
	UCS_ERR_NO_RESOURCE            UcsStatus = C.UCS_ERR_NO_RESOURCE
	UCS_ERR_IO_ERROR               UcsStatus = C.UCS_ERR_IO_ERROR
	UCS_ERR_NO_MEMORY              UcsStatus = C.UCS_ERR_NO_MEMORY
	UCS_ERR_INVALID_PARAM          UcsStatus = C.UCS_ERR_INVALID_PARAM
	UCS_ERR_UNREACHABLE            UcsStatus = C.UCS_ERR_UNREACHABLE
	UCS_ERR_INVALID_ADDR           UcsStatus = C.UCS_ERR_INVALID_ADDR
	UCS_ERR_NOT_IMPLEMENTED        UcsStatus = C.UCS_ERR_NOT_IMPLEMENTED
	UCS_ERR_MESSAGE_TRUNCATED      UcsStatus = C.UCS_ERR_MESSAGE_TRUNCATED
	UCS_ERR_NO_PROGRESS            UcsStatus = C.UCS_ERR_NO_PROGRESS
	UCS_ERR_BUFFER_TOO_SMALL       UcsStatus = C.UCS_ERR_BUFFER_TOO_SMALL
	UCS_ERR_NO_ELEM                UcsStatus = C.UCS_ERR_NO_ELEM
	UCS_ERR_SOME_CONNECTS_FAILED   UcsStatus = C.UCS_ERR_SOME_CONNECTS_FAILED
	UCS_ERR_NO_DEVICE              UcsStatus = C.UCS_ERR_NO_DEVICE
	UCS_ERR_BUSY                   UcsStatus = C.UCS_ERR_BUSY
	UCS_ERR_CANCELED               UcsStatus = C.UCS_ERR_CANCELED
	UCS_ERR_SHMEM_SEGMENT          UcsStatus = C.UCS_ERR_SHMEM_SEGMENT
	UCS_ERR_ALREADY_EXISTS         UcsStatus = C.UCS_ERR_ALREADY_EXISTS
	UCS_ERR_OUT_OF_RANGE           UcsStatus = C.UCS_ERR_OUT_OF_RANGE
	UCS_ERR_TIMED_OUT              UcsStatus = C.UCS_ERR_TIMED_OUT
	UCS_ERR_EXCEEDS_LIMIT          UcsStatus = C.UCS_ERR_EXCEEDS_LIMIT
	UCS_ERR_UNSUPPORTED            UcsStatus = C.UCS_ERR_UNSUPPORTED
	UCS_ERR_REJECTED               UcsStatus = C.UCS_ERR_REJECTED
	UCS_ERR_NOT_CONNECTED          UcsStatus = C.UCS_ERR_NOT_CONNECTED
	UCS_ERR_CONNECTION_RESET       UcsStatus = C.UCS_ERR_CONNECTION_RESET
	UCS_ERR_FIRST_LINK_FAILURE     UcsStatus = C.UCS_ERR_FIRST_LINK_FAILURE
	UCS_ERR_LAST_LINK_FAILURE      UcsStatus = C.UCS_ERR_LAST_LINK_FAILURE
	UCS_ERR_FIRST_ENDPOINT_FAILURE UcsStatus = C.UCS_ERR_FIRST_ENDPOINT_FAILURE
	UCS_ERR_ENDPOINT_TIMEOUT       UcsStatus = C.UCS_ERR_ENDPOINT_TIMEOUT
	UCS_ERR_LAST_ENDPOINT_FAILURE  UcsStatus = C.UCS_ERR_LAST_ENDPOINT_FAILURE
	UCS_ERR_LAST                   UcsStatus = C.UCS_ERR_LAST
)

func (m UcsStatus) String() string {
	return C.GoString(C.ucs_status_string(C.ucs_status_t(m)))
}
