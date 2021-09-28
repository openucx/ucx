/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"

type UcpConnectionRequest struct {
	connRequest C.ucp_conn_request_h
	listener    C.ucp_listener_h
}

func (c *UcpConnectionRequest) Reject() error {
	if status := C.ucp_listener_reject(c.listener, c.connRequest); status != C.UCS_OK {
		return NewUcxError(status)
	}
	return nil
}
