/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"
import "unsafe"

// Memory handle is an opaque object representing a memory region allocated
// through UCP library, which is optimized for remote memory access
// operations (zero-copy operations). The memory could be registered
// to one or multiple network resources that are supported by UCP,
// such as InfiniBand, Gemini, and others.
type UcpMemory struct {
	memHandle C.ucp_mem_h
	context   C.ucp_context_h
}

type UcpMemAttributes struct {

	// Address of the memory segment.
	Address unsafe.Pointer

	// Size of the memory segment.
	Length uint64

	// Type of allocated or registered memory
	MemType UcsMemoryType
}

// This routine returns address and length of memory segment mapped with
// UcpContext.MemMap routine.
func (m *UcpMemory) Query(attrs ...UcpMemAttribute) (*UcpMemAttributes, error) {
	var memAttr C.ucp_mem_attr_t

	for _, attr := range attrs {
		memAttr.field_mask |= C.ulong(attr)
	}

	if status := C.ucp_mem_query(m.memHandle, &memAttr); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	result := &UcpMemAttributes{}

	for _, attr := range attrs {
		switch attr {
		case UCP_MEM_ATTR_FIELD_ADDRESS:
			result.Address = memAttr.address
		case UCP_MEM_ATTR_FIELD_LENGTH:
			result.Length = uint64(memAttr.length)
		case UCP_MEM_ATTR_FIELD_MEM_TYPE:
			result.MemType = UcsMemoryType(memAttr.mem_type)
		}
	}

	return result, nil
}

func (m *UcpMemory) Close() error {
	if status := C.ucp_mem_unmap(m.context, m.memHandle); status != C.UCS_OK {
		return newUcxError(status)
	}

	return nil
}
