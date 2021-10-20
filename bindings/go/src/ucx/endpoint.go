/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
// #include "goucx.h"
import "C"
import (
	"unsafe"
)

type UcpEp struct {
	ep C.ucp_ep_h
}

var errorHandles = make(map[C.ucp_ep_h]UcpEpErrHandler)

func setSendParams(goRequestParams *UcpRequestParams, cRequestParams *C.ucp_request_param_t) uint64 {
	var cbId uint64
	if goRequestParams != nil {
		if goRequestParams.Cb != nil {
			cbId = register(goRequestParams.Cb)
			cRequestParams.op_attr_mask |= C.UCP_OP_ATTR_FIELD_CALLBACK | C.UCP_OP_ATTR_FIELD_USER_DATA
			cbAddr := (*C.ucp_send_nbx_callback_t)(unsafe.Pointer(&cRequestParams.cb[0]))
			*cbAddr = (C.ucp_send_nbx_callback_t)(C.ucxgo_completeGoSendRequest)
			cRequestParams.user_data = unsafe.Pointer(uintptr(cbId))
		}

		cRequestParams.SetMemType(goRequestParams)
	}

	return cbId
}

// This routine flushes all outstanding AMO and RMA communications on the endpoint.
// All the AMO and RMA operations issued on the ep prior to this call are completed
// both at the origin and at the target endpoint when this call returns.
func (e *UcpEp) FlushNonBlocking(params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t
	var cbId uint64

	setSendParams(params, &requestParams)

	request := C.ucp_ep_flush_nbx(e.ep, &requestParams)
	return NewRequest(request, cbId, nil)
}

func (e *UcpEp) CloseNonBlocking(mode C.uint, params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t
	var cbId uint64
	requestParams.op_attr_mask = C.UCP_OP_ATTR_FIELD_FLAGS
	requestParams.flags = mode

	setSendParams(params, &requestParams)

	request := C.ucp_ep_close_nbx(e.ep, &requestParams)
	delete(errorHandles, e.ep)
	return NewRequest(request, cbId, nil)
}

// Non-blocking endpoint closure. Releases the endpoint without any
// confirmation from the peer. All outstanding requests will be
// completed with UCS_ERR_CANCELED error.
func (e *UcpEp) CloseNonBlockingForce(params *UcpRequestParams) (*UcpRequest, error) {
	return e.CloseNonBlocking(C.UCP_EP_CLOSE_FLAG_FORCE, params)
}

// Non-blocking endpoint close. Schedules flushes on all outstanding operations.
func (e *UcpEp) CloseNonBlockingFlush(params *UcpRequestParams) (*UcpRequest, error) {
	return e.CloseNonBlocking(0, params)
}

// This routine sends a messages that is described by the local address and size
// to the destination endpoint. Each message is associated with a  tag value that is used for message
// matching on the UcpWorker.RecvTagNonBlocking "receiver".
// The routine is non-blocking and therefore returns immediately, however the
// actual send operation may be delayed. The send operation is considered
// completed when it is safe to reuse the source buffer.
func (e *UcpEp) SendTagNonBlocking(tag uint64, address unsafe.Pointer, size uint64,
	params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t
	var cbId uint64

	setSendParams(params, &requestParams)

	request := C.ucp_tag_send_nbx(e.ep, address, C.size_t(size), C.ucp_tag_t(tag), &requestParams)
	return NewRequest(request, cbId, nil)
}

// This routine sends an Active Message to an ep.
// Sending only header without actual data is allowed and is recommended for transferring a latency-critical amount of data.
// The maximum allowed header size can be obtained by querying worker attributes by the UcpWorker.Query() routine.
func (e *UcpEp) SendAmNonBlocking(id uint, header unsafe.Pointer, headerSize uint64,
	data unsafe.Pointer, dataSize uint64, flags UcpAmSendFlags, params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t
	var cbId uint64

	setSendParams(params, &requestParams)

	requestParams.op_attr_mask |= C.UCP_OP_ATTR_FIELD_FLAGS
	requestParams.flags = C.uint(flags)

	request := C.ucp_am_send_nbx(e.ep, C.uint(id), header, C.size_t(headerSize), data, C.size_t(dataSize), &requestParams)
	return NewRequest(request, cbId, nil)
}
