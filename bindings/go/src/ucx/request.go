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

type UcpRequest struct {
	request unsafe.Pointer
	Status  UcsStatus
}

type UcpRequestParams struct {
	MemTypeSet bool
	MemType    UcsMemoryType
	Cb         UcpCallback
}

// Checks wether request is a pointer
func isRequestPtr(request C.ucs_status_ptr_t) bool {
	errLast := UCS_ERR_LAST
	return (uint64(uintptr(request)) - 1) < (uint64(errLast) - 1)
}

func NewRequest(request C.ucs_status_ptr_t, callbackId uint64, immidiateInfo interface{}) (*UcpRequest, error) {
	ucpRequest := &UcpRequest{}

	if isRequestPtr(request) {
		ucpRequest.request = unsafe.Pointer(uintptr(request))
		ucpRequest.Status = UCS_INPROGRESS
	} else {
		requestStatus := UcsStatus(int64(uintptr(request)))
		if callback, found := deregister(callbackId); found {
			switch callback := callback.(type) {
			case UcpSendCallback:
				callback(ucpRequest, requestStatus)
			case UcpTagRecvCallback:
				callback(ucpRequest, requestStatus, immidiateInfo.(*UcpTagRecvInfo))
			}
			if requestStatus != C.UCS_OK {
				return ucpRequest, NewUcxError(C.ucs_status_t(requestStatus))
			}
		}
	}

	return ucpRequest, nil
}

// This routine checks the state of the request and returns its current status.
// Any value different from UCS_INPROGRESS means that request is in a completed
// state.
func (r *UcpRequest) GetStatus() UcsStatus {
	if r.Status != UCS_INPROGRESS {
		return r.Status
	}
	return UcsStatus(C.ucp_request_check_status(r.request))
}

// This routine releases the non-blocking request back to the library, regardless
// of its current state. Communications operations associated with this request
// will make progress internally, however no further notifications or callbacks
// will be invoked for this request.
func (r *UcpRequest) Close() {
	if r.request != nil {
		C.ucp_request_free(r.request)
	}
}
