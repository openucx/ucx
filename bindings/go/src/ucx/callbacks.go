/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include "goucx.h"
import "C"
import (
	"sync"
	"unsafe"
)

type UcpCallback interface{}

type UcpSendCallback = func(request *UcpRequest, status UcsStatus)

type UcpTagRecvCallback = func(request *UcpRequest, status UcsStatus, tagInfo *UcpTagRecvInfo)

type UcpAmDataRecvCallback = func(request *UcpRequest, status UcsStatus, length uint64)

type UcpAmRecvCallback = func(header unsafe.Pointer, headerSize uint64,
	data *UcpAmData, replyEp *UcpEp) UcsStatus

// This callback routine is invoked on the server side to handle incoming
// connections from remote clients.
type UcpListenerConnectionHandler = func(connRequest *UcpConnectionRequest)

// Map from the callback id that is passed to C to the actual go callback.
var callback_map = make(map[uint64]UcpCallback)

// Unique index for each go callback, that passes to user_data.
var callback_id uint64 = 1

var mu sync.Mutex

// Associates go callback with a unique id
func register(cb UcpCallback) uint64 {
	mu.Lock()
	defer mu.Unlock()
	callback_id++
	callback_map[callback_id] = cb
	return callback_id
}

// Atomically removes registered callback by it's id
func deregister(id uint64) (UcpCallback, bool) {
	mu.Lock()
	defer mu.Unlock()
	val, ret := callback_map[id]
	delete(callback_map, id)
	return val, ret
}

func getCallback(id uint64) (UcpCallback, bool) {
	mu.Lock()
	defer mu.Unlock()
	val, ret := callback_map[id]
	return val, ret
}

//export ucxgo_completeGoSendRequest
func ucxgo_completeGoSendRequest(request unsafe.Pointer, status C.ucs_status_t, callbackId unsafe.Pointer) {
	if callback, found := deregister(uint64(uintptr(callbackId))); found {
		callback.(UcpSendCallback)(&UcpRequest{
			request: request,
			Status:  UcsStatus(status),
		}, UcsStatus(status))
	}
}

//export ucxgo_completeGoTagRecvRequest
func ucxgo_completeGoTagRecvRequest(request unsafe.Pointer, status C.ucs_status_t, tag_info *C.ucp_tag_recv_info_t, callbackId unsafe.Pointer) {
	if callback, found := deregister(uint64(uintptr(callbackId))); found {
		callback.(UcpTagRecvCallback)(&UcpRequest{
			request: request,
			Status:  UcsStatus(status),
		}, UcsStatus(status), &UcpTagRecvInfo{
			SenderTag: uint64(tag_info.sender_tag),
			Length:    uint64(tag_info.length),
		})
	}
}

//export ucxgo_amRecvCallback
func ucxgo_amRecvCallback(calbackId unsafe.Pointer, header unsafe.Pointer, headerSize C.size_t,
	data unsafe.Pointer, dataSize C.size_t, params *C.ucp_am_recv_param_t) C.ucs_status_t {
	cbId := uint64(uintptr(calbackId))
	if callback, found := getCallback(cbId); found {
		var replyEp *UcpEp
		if (params.recv_attr & C.UCP_AM_RECV_ATTR_FIELD_REPLY_EP) != 0 {
			replyEp = &UcpEp{ep: params.reply_ep}
		}
		amData := &UcpAmData{
			worker:  idToWorker[cbId],
			flags:   UcpAmRecvAttrs(params.recv_attr),
			dataPtr: data,
			length:  uint64(dataSize),
		}
		return C.ucs_status_t(callback.(UcpAmRecvCallback)(header, uint64(headerSize), amData, replyEp))
	}
	return C.UCS_OK
}

//export ucxgo_completeAmRecvData
func ucxgo_completeAmRecvData(request unsafe.Pointer, status C.ucs_status_t,
	length C.size_t, callbackId unsafe.Pointer) {

	if callback, found := deregister(uint64(uintptr(callbackId))); found {
		callback.(UcpAmDataRecvCallback)(&UcpRequest{
			request: request,
			Status:  UcsStatus(status),
		}, UcsStatus(status), uint64(length))
	}
}
