/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
// #include "goucx.h"
import "C"
import (
	"net"
	"runtime"
	"unsafe"
)

// Tuning parameters for the UCP endpoint.
type UcpEpParams struct {
	params       C.ucp_ep_params_t
	errorHandler UcpEpErrHandler
}

// This callback routine is invoked when transport level error detected.
// ep - Endpoint to handle transport level error. Upon return
// from the callback, this endpoint is no longer usable and
// all subsequent operations on this ep will fail with
// the error code passed in status.
type UcpEpErrHandler func(ep *UcpEp, status UcsStatus)

//export ucxgo_completeGoErrorHandler
func ucxgo_completeGoErrorHandler(user_data unsafe.Pointer, ep C.ucp_ep_h, status C.ucs_status_t) {
	errHandleGoCallback, found := errorHandles[ep]
	if found {
		errHandleGoCallback(&UcpEp{
			ep: ep,
		}, UcsStatus(status))
	}
}

// Destination address
func (p *UcpEpParams) SetUcpAddress(a *UcpAddress) *UcpEpParams {
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_REMOTE_ADDRESS
	p.params.address = a.Address
	return p
}

// Guarantees that send requests are always completed (successfully or error) even in
// case of remote failure, disables protocols and APIs which may cause a hang or undefined
// behavior in case of peer failure, may affect performance and memory footprint
func (p *UcpEpParams) SetPeerErrorHandling() *UcpEpParams {
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE
	p.params.err_mode = C.ucp_err_handling_mode_t(C.UCP_ERR_HANDLING_MODE_PEER)
	return p
}

// Handler to process transport level failure.
func (p *UcpEpParams) SetErrorHandler(errHandler UcpEpErrHandler) *UcpEpParams {
	var err_handler_t C.ucp_err_handler_t
	err_handler_t.cb = (C.ucp_err_handler_cb_t)(C.ucxgo_completeGoErrorHandler)
	p.errorHandler = errHandler
	p.params.err_handler = err_handler_t
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_ERR_HANDLER
	return p
}

// Tracing and analysis tools can identify the endpoint using this name.
func (p *UcpEpParams) SetName(name string) *UcpEpParams {
	freeParamsName(p)
	p.params.name = C.CString(name)
	runtime.SetFinalizer(p, func(f *UcpEpParams) { FreeNativeMemory(unsafe.Pointer(f.params.name)) })
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_NAME
	return p
}

// Destination address in the form of a sockaddr; means
// that this type of the endpoint creation is possible only on client side
// in client-server connection establishment flow.
func (p *UcpEpParams) SetSocketAddress(a *net.TCPAddr) (*UcpEpParams, error) {
	sockAddr, error := toSockAddr(a)
	if error != nil {
		return nil, error
	}

	freeParamsAddress(p)

	p.params.sockaddr = *sockAddr
	runtime.SetFinalizer(p, func(f *UcpEpParams) { FreeNativeMemory(unsafe.Pointer(f.params.sockaddr.addr)) })
	p.params.flags |= C.UCP_EP_PARAMS_FLAGS_CLIENT_SERVER
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_SOCK_ADDR | C.UCP_EP_PARAM_FIELD_FLAGS
	return p, nil
}

// Connection request from client; means that this type of the endpoint
// creation is possible only on server side in client-server connection
// establishment flow.
func (p *UcpEpParams) SetConnRequest(c *UcpConnectionRequest) *UcpEpParams {
	p.params.conn_request = c.connRequest
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_CONN_REQUEST
	return p
}

// Send client id when connecting to remote socket address as part of the connection request payload.
// On the remote side value can be obtained by calling UcpConnectionRequest.Query(UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID)
func (p *UcpEpParams) SendClientId() *UcpEpParams {
	p.params.flags |= C.UCP_EP_PARAMS_FLAGS_SEND_CLIENT_ID
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_FLAGS
	return p
}
