/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"
import "net"

type UcpListener struct {
	listener      C.ucp_listener_h
	connHandlerId uint64
}

// Needed to call connHandler.Reject() rather than listener.Reject(connHandler)
var connHandles2Listener = make(map[uint64]C.ucp_listener_h)

type UcpListenerAttributes struct {
	Address *net.TCPAddr
}

func (l *UcpListener) Close() {
	C.ucp_listener_destroy(l.listener)
	deregister(l.connHandlerId)
	delete(connHandles2Listener, l.connHandlerId)
}

func (l *UcpListener) Query(attrs ...UcpListenerAttribute) (*UcpListenerAttributes, error) {
	var listenerAttr C.ucp_listener_attr_t

	for _, attr := range attrs {
		listenerAttr.field_mask |= C.ulong(attr)
	}

	if status := C.ucp_listener_query(l.listener, &listenerAttr); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	result := &UcpListenerAttributes{}

	for _, attr := range attrs {
		switch attr {
		case UCP_LISTENER_ATTR_FIELD_SOCKADDR:
			result.Address = toTcpAddr(&listenerAttr.sockaddr)
		}
	}
	return result, nil
}
