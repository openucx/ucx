/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucp

// #include <ucp/api/ucp.h>
// #include <ucs/type/status.h>
import "C"

import (
	"fmt"
)

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

func NewUcpContext(params *UcpParams) (*UcpContext, error) {
	var ucp_context C.ucp_context_h

	if status := C.ucp_init(&params.params, nil, &ucp_context); status != C.UCS_OK {
		return nil, fmt.Errorf("unable to init context: %s", C.GoString(C.ucs_status_string(status)))
	}

	ctx := &UcpContext{
		context: ucp_context,
	}
	return ctx, nil
}

func (p *UcpContext) Close() error {
	C.ucp_cleanup(p.context)
	p.context = nil
	return nil
}
