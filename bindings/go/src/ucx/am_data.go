/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"
import (
	"errors"
	"unsafe"
)

// Active Message data descriptor
type UcpAmData struct {
	worker  *UcpWorker
	dataPtr unsafe.Pointer
	length  uint64
	flags   UcpAmRecvAttrs
}

// To connect callback id with worker, to use in AmData.Receive()
var idToWorker = make(map[uint64]*UcpWorker)

// Whether actual data is received or need to call UcpAmData.Receive()
func (d *UcpAmData) IsDataValid() bool {
	return (d.flags & UCP_AM_RECV_ATTR_FLAG_RNDV) == 0
}

// Whether this amData descriptor can be persisted outside UcpAmRecvCallback
// callback by returning UCS_INPROGRESS
func (d *UcpAmData) CanPersist() bool {
	return (d.flags & UCP_AM_RECV_ATTR_FLAG_DATA) != 0
}

// Pointer to a received data
func (d *UcpAmData) DataPointer() (unsafe.Pointer, error) {
	if !d.IsDataValid() {
		return nil, errors.New("data is not received yet")
	}
	return d.dataPtr, nil
}

func (d *UcpAmData) Length() uint64 {
	return d.length
}

func (d *UcpAmData) Receive(recvBuffer unsafe.Pointer, size uint64, params *UcpRequestParams) (*UcpRequest, error) {
	return d.worker.RecvAmDataNonBlocking(d, recvBuffer, size, params)
}

func (d *UcpAmData) Close() {
	C.ucp_am_data_release(d.worker.worker, d.dataPtr)
}
