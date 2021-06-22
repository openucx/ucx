/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucp

// #include <ucp/api/ucp.h>
import "C"

// Memory handle is an opaque object representing a memory region allocated
// through UCP library, which is optimized for remote memory access
// operations (zero-copy operations). The memory could be registered
// to one or multiple network resources that are supported by UCP,
// such as InfiniBand, Gemini, and others.
type UcpMemory struct {
	memHandle C.ucp_mem_h
	context   C.ucp_context_h
}

// This routine returns address and length of memory segment mapped with
// UcpContext.MemMap routine.
func (m *UcpMemory) Query(attrs ...UcpMemAttribute) (*C.ucp_mem_attr_t, error) {
	var memAttr C.ucp_mem_attr_t

	for attr, _ := range attrs {
		memAttr.field_mask |= C.ulong(attr)
	}

	if status := C.ucp_mem_query(m.memHandle, &memAttr); status != C.UCS_OK {
		return nil, NewUcxError(status)
	}

	return &memAttr, nil
}

func (m *UcpMemory) Close() error {
	if status := C.ucp_mem_unmap(m.context, m.memHandle); status != C.UCS_OK {
		return NewUcxError(status)
	}

	return nil
}
