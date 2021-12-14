/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"
import (
	"net"
)

type UcpConnectionRequest struct {
	connRequest C.ucp_conn_request_h
	listener    C.ucp_listener_h
}

type UcpConnectionRequestAttributes struct {
	ClientAddress *net.TCPAddr
	ClientId      uint64
}

func (c *UcpConnectionRequest) Reject() error {
	if status := C.ucp_listener_reject(c.listener, c.connRequest); status != C.UCS_OK {
		return newUcxError(status)
	}
	return nil
}

func (c *UcpConnectionRequest) Query(attrs ...UcpConnRequestAttribute) (*UcpConnectionRequestAttributes, error) {
	var connReqAttr C.ucp_conn_request_attr_t

	for _, attr := range attrs {
		connReqAttr.field_mask |= C.ulong(attr)
	}

	if status := C.ucp_conn_request_query(c.connRequest, &connReqAttr); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	result := &UcpConnectionRequestAttributes{}

	for _, attr := range attrs {
		switch attr {
		case UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR:
			result.ClientAddress = toTcpAddr(&connReqAttr.client_address)
		case UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID:
			result.ClientId = uint64(connReqAttr.client_id)
		}
	}
	return result, nil
}
