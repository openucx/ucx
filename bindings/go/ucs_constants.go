/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucp

// #include <ucp/api/ucp.h>
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

// Checks whether context's memory type mask
// (received via UcpContext.MemoryTypesMask()) supports particular memory type.
func IsMemTypeSupported(memType UcsMemoryType, mask uint64) bool {
	return ((1 << memType) & mask) != 0
}

type UcxStatus int

const (
	UCS_OK                         UcxStatus = C.UCS_OK
	UCS_INPROGRESS                 UcxStatus = C.UCS_INPROGRESS
	UCS_ERR_NO_MESSAGE             UcxStatus = C.UCS_ERR_NO_MESSAGE
	UCS_ERR_NO_RESOURCE            UcxStatus = C.UCS_ERR_NO_RESOURCE
	UCS_ERR_IO_ERROR               UcxStatus = C.UCS_ERR_IO_ERROR
	UCS_ERR_NO_MEMORY              UcxStatus = C.UCS_ERR_NO_MEMORY
	UCS_ERR_INVALID_PARAM          UcxStatus = C.UCS_ERR_INVALID_PARAM
	UCS_ERR_UNREACHABLE            UcxStatus = C.UCS_ERR_UNREACHABLE
	UCS_ERR_INVALID_ADDR           UcxStatus = C.UCS_ERR_INVALID_ADDR
	UCS_ERR_NOT_IMPLEMENTED        UcxStatus = C.UCS_ERR_NOT_IMPLEMENTED
	UCS_ERR_MESSAGE_TRUNCATED      UcxStatus = C.UCS_ERR_MESSAGE_TRUNCATED
	UCS_ERR_NO_PROGRESS            UcxStatus = C.UCS_ERR_NO_PROGRESS
	UCS_ERR_BUFFER_TOO_SMALL       UcxStatus = C.UCS_ERR_BUFFER_TOO_SMALL
	UCS_ERR_NO_ELEM                UcxStatus = C.UCS_ERR_NO_ELEM
	UCS_ERR_SOME_CONNECTS_FAILED   UcxStatus = C.UCS_ERR_SOME_CONNECTS_FAILED
	UCS_ERR_NO_DEVICE              UcxStatus = C.UCS_ERR_NO_DEVICE
	UCS_ERR_BUSY                   UcxStatus = C.UCS_ERR_BUSY
	UCS_ERR_CANCELED               UcxStatus = C.UCS_ERR_CANCELED
	UCS_ERR_SHMEM_SEGMENT          UcxStatus = C.UCS_ERR_SHMEM_SEGMENT
	UCS_ERR_ALREADY_EXISTS         UcxStatus = C.UCS_ERR_ALREADY_EXISTS
	UCS_ERR_OUT_OF_RANGE           UcxStatus = C.UCS_ERR_OUT_OF_RANGE
	UCS_ERR_TIMED_OUT              UcxStatus = C.UCS_ERR_TIMED_OUT
	UCS_ERR_EXCEEDS_LIMIT          UcxStatus = C.UCS_ERR_EXCEEDS_LIMIT
	UCS_ERR_UNSUPPORTED            UcxStatus = C.UCS_ERR_UNSUPPORTED
	UCS_ERR_REJECTED               UcxStatus = C.UCS_ERR_REJECTED
	UCS_ERR_NOT_CONNECTED          UcxStatus = C.UCS_ERR_NOT_CONNECTED
	UCS_ERR_CONNECTION_RESET       UcxStatus = C.UCS_ERR_CONNECTION_RESET
	UCS_ERR_FIRST_LINK_FAILURE     UcxStatus = C.UCS_ERR_FIRST_LINK_FAILURE
	UCS_ERR_LAST_LINK_FAILURE      UcxStatus = C.UCS_ERR_LAST_LINK_FAILURE
	UCS_ERR_FIRST_ENDPOINT_FAILURE UcxStatus = C.UCS_ERR_FIRST_ENDPOINT_FAILURE
	UCS_ERR_ENDPOINT_TIMEOUT       UcxStatus = C.UCS_ERR_ENDPOINT_TIMEOUT
	UCS_ERR_LAST_ENDPOINT_FAILURE  UcxStatus = C.UCS_ERR_LAST_ENDPOINT_FAILURE
	UCS_ERR_LAST                   UcxStatus = C.UCS_ERR_LAST
)
