/*
 * Copyright (C) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"
import "unsafe"

type UcpRKey struct {
	rkey      C.ucp_rkey_h
}

type UcpRKeyBuffer struct {
	buffer    unsafe.Pointer
	size      C.size_t
}

func NewRKeyBuffer(buffer []byte) *UcpRKeyBuffer {
	return &UcpRKeyBuffer{
		buffer: unsafe.Pointer(&buffer[0]),
		size:   C.size_t(len(buffer)),
	}
}

func (m *UcpMemory) Pack() (*UcpRKeyBuffer, error) {
	result := &UcpRKeyBuffer{}

	if status := C.ucp_rkey_pack(m.context, m.memHandle, &result.buffer, &result.size); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	return result, nil
}

func (b *UcpRKeyBuffer) Bytes() []byte {
	return unsafe.Slice((*byte)(b.buffer), b.size)
}

func (b *UcpRKeyBuffer) Close() {
	var releaseParam C.ucp_memh_buffer_release_params_t
	C.ucp_memh_buffer_release(b.buffer, &releaseParam)
}

func (e *UcpEp) Unpack(buffer *UcpRKeyBuffer) (*UcpRKey, error) {
	result := &UcpRKey{}
	if status := C.ucp_ep_rkey_unpack(e.ep, buffer.buffer, &result.rkey); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	return result, nil
}

func (r *UcpRKey) Close() {
	C.ucp_rkey_destroy(r.rkey)
}

func (e *UcpEp) RmaPut(buffer unsafe.Pointer, size uint64, remote_addr uint64, rkey *UcpRKey, params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t

	cbId := setSendParams(params, &requestParams)

	request := C.ucp_put_nbx(e.ep, buffer, C.size_t(size), C.uint64_t(remote_addr), rkey.rkey, &requestParams)

	return NewRequest(request, cbId, nil)
}

func (e *UcpEp) RmaGet(buffer unsafe.Pointer, size uint64, remote_addr uint64, rkey *UcpRKey, params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t

	cbId := setSendParams(params, &requestParams)

	request := C.ucp_get_nbx(e.ep, buffer, C.size_t(size), C.uint64_t(remote_addr), rkey.rkey, &requestParams)

	return NewRequest(request, cbId, nil)
}
