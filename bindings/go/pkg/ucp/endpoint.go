package ucp

/*
#include <ucp/api/ucp.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type endpointContext struct {
	endpointCtx C.ucp_ep_h
	workerCtx   C.ucp_worker_h
	params      C.ucp_ep_params_t
}

func newEndpoint(workerCtx *workerContext, params *endpointParams) (*endpointContext, error) {
	var ucp_ep C.ucp_ep_h

	if status := C.ucp_ep_create(workerCtx.workerCtx, &params.params, &ucp_ep); status != C.UCS_OK {
		return nil, fmt.Errorf("Unable to create endpoint: %d", status)
	}

	endpointCtx := &endpointContext{
		endpointCtx: ucp_ep,
		workerCtx:   workerCtx.workerCtx,
		params:      params.params}
	return endpointCtx, nil
}

func (c *endpointContext) Close() error {
	C.ucp_ep_destroy(c.endpointCtx)
	return nil
}

func (c *endpointContext) UnpackRemoteKey(b []byte) (C.ucp_rkey_h, error) {
	var rkey C.ucp_rkey_h

	if status := C.ucp_ep_rkey_unpack(c.endpointCtx, unsafe.Pointer(&b[0]), &rkey); status != C.UCS_OK {
		return nil, fmt.Errorf("Unable to unpack remote key: %d", status)
	}
	return rkey, nil
}

/*
func (c *endpointContext) PutNonBlocking(local, remote []byte, remoteKey C.ucp_rkey_h, UcxCallback callback) C.ucs_status_ptr_t {
	return C.putNonBlockingNative(c.endpointCtx, unsafe.Pointer(&local[0]), len(local), unsafe.Pointer(&remote[0]), remoteKey, callback)
}
*/
