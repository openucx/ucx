/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"
import "unsafe"

// Tuning parameters for the UCP memory mapping.
type UcpMmapParams struct {
	params C.ucp_mem_map_params_t
}

// If the address is not NULL, the routine maps (registers) the memory segment
// pointed to by this address.
// If the pointer is NULL, the library allocates mapped (registered) memory
// segment and returns its address in this argument.
func (p *UcpMmapParams) SetAddress(address unsafe.Pointer) *UcpMmapParams {
	p.params.address = address
	p.params.field_mask |= C.UCP_MEM_MAP_PARAM_FIELD_ADDRESS
	return p
}

// Length (in bytes) to allocate or map (register).
func (p *UcpMmapParams) SetLength(length uint64) *UcpMmapParams {
	p.params.length = C.size_t(length)
	p.params.field_mask |= C.UCP_MEM_MAP_PARAM_FIELD_LENGTH
	return p
}

// Identify requirement for allocation, if passed address is not a null-pointer
// then it will be used as a hint or direct address for allocation.
func (p *UcpMmapParams) Allocate() *UcpMmapParams {
	p.params.field_mask |= C.UCP_MEM_MAP_PARAM_FIELD_FLAGS
	p.params.flags |= C.UCP_MEM_MAP_ALLOCATE
	return p
}

// Complete the registration faster, possibly by not populating the pages up-front,
// and mapping them later when they are accessed by communication routines.
func (p *UcpMmapParams) Nonblocking() *UcpMmapParams {
	p.params.field_mask |= C.UCP_MEM_MAP_PARAM_FIELD_FLAGS
	p.params.flags |= C.UCP_MEM_MAP_NONBLOCK
	return p
}

// Don't interpret address as a hint: place the mapping at exactly that
// address. The address must be a multiple of the page size.
func (p *UcpMmapParams) Fixed() *UcpMmapParams {
	p.params.field_mask |= C.UCP_MEM_MAP_PARAM_FIELD_FLAGS
	p.params.flags |= C.UCP_MEM_MAP_FIXED
	return p
}

// Memory protection mode, e.g. UCP_MEM_MAP_PROT_LOCAL_READ
// This value is optional. If it's not set, the UcpContext.Mmap
// routine will consider the flags as set to
// UCP_MEM_MAP_PROT_LOCAL_READ|UCP_MEM_MAP_PROT_LOCAL_WRITE|
// UCP_MEM_MAP_PROT_REMOTE_READ|UCP_MEM_MAP_PROT_REMOTE_WRITE.
func (p *UcpMmapParams) SetProtection(prot UcpProtection) *UcpMmapParams {
	p.params.field_mask |= C.UCP_MEM_MAP_PARAM_FIELD_PROT
	p.params.prot = C.uint(prot)
	return p
}

// Memory type (for possible memory types see UcsMemoryType)
// It is an optimization hint to avoid memory type detection for map buffer.
// The meaning of this field depends on the operation type.
//
// - Memory allocation: (UcpMmapParams.Allocate() is set) This field
//   specifies the type of memory to allocate. If it's not set
//   UCS_MEMORY_TYPE_HOST will be assumed by default.
//
// - Memory registration: This field specifies the type of memory which is
//   pointed by UcpMmapParams.SetAddress(). If it's not set,
//   or set to UCS_MEMORY_TYPE_UNKNOWN, the memory type will be detected internally.
//
func (p *UcpMmapParams) SetMemoryType(memType UcsMemoryType) *UcpMmapParams {
	p.params.field_mask |= C.UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE
	p.params.memory_type = C.ucs_memory_type_t(memType)
	return p
}
