/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
// #include <ucs/type/status.h>
import "C"

// UCP application context (or just a context) is an opaque handle that holds a
// UCP communication instance's global information. It represents a single UCP
// communication instance. The communication instance could be an OS process
// (an application) that uses UCP library.  This global information includes
// communication resources, endpoints, memory, temporary file storage, and
// other communication information directly associated with a specific UCP
// instance. The context also acts as an isolation mechanism, allowing
// resources associated with the context to manage multiple concurrent
// communication instances. For example, users can isolate their communication
// by allocating and using separate contexts. Alternatively, users can share the
// communication resources (memory, network resource context, etc.) between
// them by using the same application context. A message sent or a RMA
// operation performed in one application context cannot be received in any
// other application context.
type UcpContext struct {
	context C.ucp_context_h
}

func NewUcpContext(contextParams *UcpParams) (*UcpContext, error) {
	var ucp_context C.ucp_context_h

	if status := C.ucp_init(&contextParams.params, nil, &ucp_context); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	ctx := &UcpContext{
		context: ucp_context,
	}
	return ctx, nil
}

func (c *UcpContext) Close() error {
	C.ucp_cleanup(c.context)
	c.context = nil
	return nil
}

// Mask which memory types are supported
func (c *UcpContext) MemoryTypesMask() (uint64, error) {
	ucp_attrs, err := c.Query(UCP_ATTR_FIELD_MEMORY_TYPES)
	if err != nil {
		return 0, err
	}
	return uint64(ucp_attrs.memory_types), nil
}

// Associates memory allocated/mapped region with communication operations
// The network stack associated with an application context
// can typically send and receive data from the mapped memory without
// CPU intervention; some devices and associated network stacks
// require the memory to be registered to send and receive data.
func (c *UcpContext) MemMap(memMapParams *UcpMmapParams) (*UcpMemory, error) {
	var ucp_memh C.ucp_mem_h

	if status := C.ucp_mem_map(c.context, &memMapParams.params, &ucp_memh); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	return &UcpMemory{
		memHandle: ucp_memh,
		context:   c.context,
	}, nil
}

// This routine fetches information about the context.
func (c *UcpContext) Query(attrs ...UcpContextAttr) (*C.ucp_context_attr_t, error) {
	var ucp_attrs C.ucp_context_attr_t

	for _, attr := range attrs {
		ucp_attrs.field_mask |= C.ulong(attr)
	}

	if status := C.ucp_context_query(c.context, &ucp_attrs); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	return &ucp_attrs, nil
}

// This routine creates new UcpWorker.
func (c *UcpContext) NewWorker(workerParams *UcpWorkerParams) (*UcpWorker, error) {
	var ucp_worker C.ucp_worker_h

	if status := C.ucp_worker_create(c.context, &workerParams.params, &ucp_worker); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	return &UcpWorker{
		worker: ucp_worker,
	}, nil
}
