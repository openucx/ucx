/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include "goucx.h"
import "C"
import (
	"unsafe"
	"runtime/cgo"
)

type UcpCallback interface{}

type UcpSendCallback = func(request *UcpRequest, status UcsStatus)

type UcpTagRecvCallback = func(request *UcpRequest, status UcsStatus, tagInfo *UcpTagRecvInfo)

type UcpAmDataRecvCallback = func(request *UcpRequest, status UcsStatus, length uint64)

type UcpAmRecvCallback = func(header unsafe.Pointer, headerSize uint64,
	data *UcpAmData, replyEp *UcpEp) UcsStatus

type UcpAmRecvCallbackBundle struct {
	cb     UcpAmRecvCallback
	worker *UcpWorker
}	

// This callback routine is invoked on the server side to handle incoming
// connections from remote clients.
type UcpListenerConnectionHandler = func(connRequest *UcpConnectionRequest)

type PackedCallback cgo.Handle
func packCallback(cb UcpCallback) PackedCallback {
    if cb == nil {
        return 0
    }

    return PackedCallback(cgo.NewHandle(cb))
}

func (pc PackedCallback) unpackInternal(freeHandle bool) UcpCallback {
    if pc == 0 {
        return nil
    }

    h := cgo.Handle(pc)
    if freeHandle {
        defer h.Delete()
    }

    return h.Value().(UcpCallback)
}

func (pc PackedCallback) unpack() UcpCallback {
    return pc.unpackInternal(false)
}

func (pc PackedCallback) unpackAndFree() UcpCallback {
    return pc.unpackInternal(true)
}

//export ucxgo_completeGoSendRequest
func ucxgo_completeGoSendRequest(request unsafe.Pointer, status C.ucs_status_t, packedCb unsafe.Pointer) {
	if callback := PackedCallback(packedCb).unpackAndFree(); callback != nil {
		callback.(UcpSendCallback)(&UcpRequest{
			request: request,
			Status:  UcsStatus(status),
		}, UcsStatus(status))
	}
}

//export ucxgo_completeGoTagRecvRequest
func ucxgo_completeGoTagRecvRequest(request unsafe.Pointer, status C.ucs_status_t, tag_info *C.ucp_tag_recv_info_t, packedCb unsafe.Pointer) {
	if callback := PackedCallback(packedCb).unpackAndFree(); callback != nil {
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
func ucxgo_amRecvCallback(packedCb unsafe.Pointer, header unsafe.Pointer, headerSize C.size_t,
	data unsafe.Pointer, dataSize C.size_t, params *C.ucp_am_recv_param_t) C.ucs_status_t {
	if callback := PackedCallback(packedCb).unpack(); callback != nil {
		bundle := callback.(*UcpAmRecvCallbackBundle)
		var replyEp *UcpEp
		if (params.recv_attr & C.UCP_AM_RECV_ATTR_FIELD_REPLY_EP) != 0 {
			replyEp = &UcpEp{ep: params.reply_ep}
		}
		amData := &UcpAmData{
			worker:  bundle.worker,
			flags:   UcpAmRecvAttrs(params.recv_attr),
			dataPtr: data,
			length:  uint64(dataSize),
		}
		return C.ucs_status_t(bundle.cb(header, uint64(headerSize), amData, replyEp))
	}
	return C.UCS_OK
}

//export ucxgo_completeAmRecvData
func ucxgo_completeAmRecvData(request unsafe.Pointer, status C.ucs_status_t,
	length C.size_t, packedCb unsafe.Pointer) {

	if callback := PackedCallback(packedCb).unpackAndFree(); callback != nil {
		callback.(UcpAmDataRecvCallback)(&UcpRequest{
			request: request,
			Status:  UcsStatus(status),
		}, UcsStatus(status), uint64(length))
	}
}
