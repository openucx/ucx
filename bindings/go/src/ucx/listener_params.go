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

// Tuning parameters for the UCP listener.
type UcpListenerParams struct {
	params        C.ucp_listener_params_t
	connHandlerId uint64
}

//export ucxgo_completeConnHandler
func ucxgo_completeConnHandler(connRequest C.ucp_conn_request_h, cbId unsafe.Pointer) {
	id := uint64(uintptr((cbId)))
	if callback, found := getCallback(id); found {
		listener := connHandles2Listener[id]
		callback.(UcpListenerConnectionHandler)(&UcpConnectionRequest{
			connRequest: connRequest,
			listener:    listener,
		})
	}
}

// Destination address
func (p *UcpListenerParams) SetSocketAddress(a *net.TCPAddr) (*UcpListenerParams, error) {
	sockAddr, error := toSockAddr(a)
	if error != nil {
		return nil, error
	}

	freeParamsAddress(p)
	p.params.field_mask |= C.UCP_LISTENER_PARAM_FIELD_SOCK_ADDR
	p.params.sockaddr = *sockAddr
	runtime.SetFinalizer(p, func(f *UcpListenerParams) { FreeNativeMemory(unsafe.Pointer(f.params.sockaddr.addr)) })
	return p, nil
}

// Handler of an incoming connection request in a client-server connection flow.
func (p *UcpListenerParams) SetConnectionHandler(connHandler UcpListenerConnectionHandler) *UcpListenerParams {
	var ucpConnHndl C.ucp_listener_conn_handler_t
	cbId := register(connHandler)

	p.connHandlerId = cbId
	ucpConnHndl.arg = unsafe.Pointer(uintptr(cbId))
	ucpConnHndl.cb = (C.ucp_listener_conn_callback_t)(C.ucxgo_completeConnHandler)
	p.params.field_mask |= C.UCP_LISTENER_PARAM_FIELD_CONN_HANDLER
	p.params.conn_handler = ucpConnHndl
	return p
}
