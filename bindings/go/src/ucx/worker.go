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

// UCP worker is an opaque object representing the communication context. The
// worker represents an instance of a local communication resource and the
// progress engine associated with it. The progress engine is a construct that
// is responsible for asynchronous and independent progress of communication
// directives. The progress engine could be implemented in hardware or software.
// The worker object abstracts an instance of network resources such as a host
// channel adapter port, network interface, or multiple resources such as
// multiple network interfaces or communication ports. It could also represent
// virtual communication resources that are defined across multiple devices.
// Although the worker can represent multiple network resources, it is
// associated with a single UcpContext "UCX application context".
// All communication functions require a context to perform the operation on
// the dedicated hardware resource(s) and an "endpoint" to address the
// destination.
//
// Worker are parallel "threading points" that an upper layer may use to
// optimize concurrent communications.
type UcpWorker struct {
	worker C.ucp_worker_h
}

type UcpAddress struct {
	worker  C.ucp_worker_h
	Address *C.ucp_address_t
	Length  uint64
}

type UcpTagRecvInfo struct {
	SenderTag uint64
	Length    uint64
}

type UcpWorkerAttributes struct {
	ThreadMode     UcsThreadMode
	Address        *UcpAddress
	MaxAmHeader    uint64
	MaxDebugString uint64
}

func (w *UcpWorker) Close() {
	C.ucp_worker_destroy(w.worker)
}

func (a *UcpAddress) Close() {
	C.ucp_worker_release_address(a.worker, a.Address)
}

func (w *UcpWorker) Query(attrs ...UcpWorkerAttribute) (*UcpWorkerAttributes, error) {
	var workerAttr C.ucp_worker_attr_t

	for _, attr := range attrs {
		workerAttr.field_mask |= C.ulong(attr)
	}

	if status := C.ucp_worker_query(w.worker, &workerAttr); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	result := &UcpWorkerAttributes{}

	for _, attr := range attrs {
		switch attr {
		case UCP_WORKER_ATTR_FIELD_THREAD_MODE:
			result.ThreadMode = UcsThreadMode(workerAttr.thread_mode)
		case UCP_WORKER_ATTR_FIELD_ADDRESS:
			result.Address = &UcpAddress{
				worker:  w.worker,
				Address: workerAttr.address,
				Length:  uint64(workerAttr.address_length),
			}
		case UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER:
			result.MaxAmHeader = uint64(workerAttr.max_am_header)
		case UCP_WORKER_ATTR_FIELD_MAX_INFO_STRING:
			result.MaxDebugString = uint64(workerAttr.max_debug_string)
		}
	}

	return result, nil
}

// This routine needs to be called before waiting on each notification on this
// worker, so will typically be called once the processing of the previous event
// is over, as part of the wake-up mechanism.
//
// The worker must be armed before waiting on an event (must be re-armed after
// it has been signaled for re-use) with UcpWorker.Arm().
// The events triggering a signal of the file descriptor from
// UcpWorker.GetEfd() depend on the interfaces used by the worker and
// defined in the transport layer, and typically represent a request completion
// or newly available resources. It can also be triggered by calling
// UcpWorker.Signal().
//
// The file descriptor is guaranteed to become signaled only if new communication
// events occur on the worker. Therefore one must drain all existing events
// before waiting on the file descriptor. This can be achieved by calling
// UcpWorker.Progress() repeatedly until it returns 0.
func (w *UcpWorker) Arm() UcsStatus {
	return UcsStatus(C.ucp_worker_arm(w.worker))
}

// This routine explicitly progresses all communication operations on a worker.
// Typically, request wait and test routines call UcpWorker.Progress()
// "this routine" to progress any outstanding operations.
// Transport layers, implementing asynchronous progress using threads,
// require callbacks and other user code to be thread safe.
// The state of communication can be advanced (progressed) by blocking
// routines. Nevertheless, the non-blocking routines can not be used for
// communication progress.
func (w *UcpWorker) Progress() uint {
	return uint(C.ucp_worker_progress(w.worker))
}

// This routine waits (blocking) until an event has happened, as part of the
// wake-up mechanism.
//
// This function is guaranteed to return only if new communication events occur
// on the UcpWorker. Therefore one must drain all existing events before waiting
// on the file descriptor. This can be achieved by calling
// UcpWorker.Progress() repeatedly until it returns 0.
//
// There are two alternative ways to use the wakeup mechanism. The first is by
// polling on a per-worker file descriptor obtained from UcpWorker.GetEfd().
// The second is by using this function to perform an internal wait for the next
// event associated with the specified worker.
//
// @note During the blocking call the wake-up mechanism relies on other means of
// notification and may not progress some of the requests as it would when
// calling UcpWorker.Progress() (which is not invoked in that duration).
func (w *UcpWorker) Wait() error {
	if status := C.ucp_worker_wait(w.worker); status != C.UCS_OK {
		return newUcxError(status)
	}
	return nil
}

// This routine returns a valid file descriptor for polling functions.
// The file descriptor will get signaled when an event occurs, as part of the
// wake-up mechanism. Signaling means a call to poll() or select() with this
// file descriptor will return at this point, with this descriptor marked as the
// reason (or one of the reasons) the function has returned. The user does not
// need to release the obtained file descriptor.
//
// The wake-up mechanism exists to allow for the user process to register for
// notifications on events of the underlying interfaces, and wait until such
// occur. This is an alternative to repeated polling for request completion.
// The goal is to allow for waiting while consuming minimal resources from the
// system. This is recommended for cases where traffic is infrequent, and
// latency can be traded for lower resource consumption while waiting for it.
//
// There are two alternative ways to use the wakeup mechanism: the first is the
// file descriptor obtained per worker (this function) and the second is the
// UcpWorker.Wait() function for waiting on the next event internally.
func (w *UcpWorker) GetEfd() (int, error) {
	var efd C.int
	if status := C.ucp_worker_get_efd(w.worker, &efd); status != C.UCS_OK {
		return 0, newUcxError(status)
	}
	return int(efd), nil
}

// This routine signals that the event has happened, as part of the wake-up
// mechanism. This function causes a blocking call to UcpWorker.Wait() or
// waiting on a file descriptor from UcpWorker.GetEfd() to return, even
// if no event from the underlying interfaces has taken place.
func (w *UcpWorker) Signal() error {
	if status := C.ucp_worker_signal(w.worker); status != C.UCS_OK {
		return newUcxError(status)
	}
	return nil
}

// This routine returns the address of the worker object. This address can be
// passed to remote instances of the UCP library in order to connect to this
// worker. Ucp worker address - is an opaque object that is used as an
// identifier for a UcpWorker instance.
func (w *UcpWorker) GetAddress() (*UcpAddress, error) {
	result, err := w.Query(UCP_WORKER_ATTR_FIELD_ADDRESS)

	if err != nil {
		return nil, err
	}

	return result.Address, nil
}

// This routine creates new UcpEndpoint.
func (w *UcpWorker) NewEndpoint(epParams *UcpEpParams) (*UcpEp, error) {
	var ep C.ucp_ep_h

	if status := C.ucp_ep_create(w.worker, &epParams.params, &ep); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	if epParams.errorHandler != nil {
		errorHandles[ep] = epParams.errorHandler
	}

	return &UcpEp{
		ep: ep,
	}, nil
}

// This routine receives a message that is described by the local address and size on the worker.
// The tag value of the receive message has to match thetag and tagMask values,
// where the tagMask indicates what bits of the tag have to be matched. The
// routine is a non-blocking and therefore returns immediately. The receive
// operation is considered completed when the message is delivered to the buffer.
// In order to notify the application about completion of the receive
// operation the UCP library will invoke the call-back when the received
// message is in the receive buffer and ready for application access.  If the
// receive operation cannot be stated the routine returns an error.
func (w *UcpWorker) RecvTagNonBlocking(address unsafe.Pointer, size uint64,
	tag uint64, tagMask uint64, params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t
	var recvInfo C.ucp_tag_recv_info_t
	var cbId uint64

	requestParams.op_attr_mask = C.UCP_OP_ATTR_FIELD_RECV_INFO
	recvInfoPtr := (*C.ucp_tag_recv_info_t)(unsafe.Pointer(&requestParams.recv_info[0]))
	*recvInfoPtr = recvInfo

	if params != nil {
		(&requestParams).SetMemType(params)

		if params.Cb != nil {
			cbId = register(params.Cb)
			requestParams.op_attr_mask |= C.UCP_OP_ATTR_FIELD_CALLBACK | C.UCP_OP_ATTR_FIELD_USER_DATA
			cbAddr := (*C.ucp_tag_recv_nbx_callback_t)(unsafe.Pointer(&requestParams.cb[0]))
			*cbAddr = (C.ucp_tag_recv_nbx_callback_t)(C.ucxgo_completeGoTagRecvRequest)
			requestParams.user_data = unsafe.Pointer(uintptr(cbId))
		}
	}

	request := C.ucp_tag_recv_nbx(w.worker, address, C.size_t(size), C.ucp_tag_t(tag),
		C.ucp_tag_t(tagMask), &requestParams)

	return NewRequest(request, cbId, &UcpTagRecvInfo{
		SenderTag: uint64(recvInfo.sender_tag),
		Length:    uint64(recvInfo.length),
	})
}

// This routine creates new UcpListener.
func (w *UcpWorker) NewListener(listenerParams *UcpListenerParams) (*UcpListener, error) {
	var listener C.ucp_listener_h

	if status := C.ucp_listener_create(w.worker, &listenerParams.params, &listener); status != C.UCS_OK {
		return nil, newUcxError(status)
	}

	connHandles2Listener[listenerParams.connHandlerId] = listener

	return &UcpListener{listener, listenerParams.connHandlerId}, nil
}

// This routine installs a user defined callback to handle incoming Active
// Messages with a specific id. This callback is called whenever an Active
// Message that was sent from the remote peer by UcpEndpoint.SendAm is
// received on this worker.
func (w *UcpWorker) SetAmRecvHandler(id uint, flags UcpAmCbFlags, cb UcpAmRecvCallback) error {
	var amHandlerParams C.ucp_am_handler_param_t
	cbId := register(cb)
	idToWorker[cbId] = w

	amHandlerParams.field_mask = C.UCP_AM_HANDLER_PARAM_FIELD_ID |
		C.UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
		C.UCP_AM_HANDLER_PARAM_FIELD_CB |
		C.UCP_AM_HANDLER_PARAM_FIELD_ARG
	amHandlerParams.id = C.uint(id)
	amHandlerParams.arg = unsafe.Pointer(uintptr(cbId))
	amHandlerParams.flags = C.uint32_t(flags)
	cbAddr := (*C.ucp_am_recv_callback_t)(unsafe.Pointer(&amHandlerParams.cb))
	*cbAddr = (C.ucp_am_recv_callback_t)(C.ucxgo_amRecvCallback)

	status := C.ucp_worker_set_am_recv_handler(w.worker, &amHandlerParams)
	if status != C.UCS_OK {
		return newUcxError(status)
	}

	return nil
}

// Receive Active Message as defined by provided data descriptor.
func (w *UcpWorker) RecvAmDataNonBlocking(dataDesc *UcpAmData, recvBuffer unsafe.Pointer, size uint64,
	params *UcpRequestParams) (*UcpRequest, error) {
	var requestParams C.ucp_request_param_t
	var cbId uint64
	var length C.size_t

	requestParams.op_attr_mask = C.UCP_OP_ATTR_FIELD_RECV_INFO
	recvInfoPtr := (**C.size_t)(unsafe.Pointer(&requestParams.recv_info[0]))
	*recvInfoPtr = &length

	if params != nil {
		(&requestParams).SetMemType(params)

		if params.Cb != nil {
			cbId = register(params.Cb)
			requestParams.op_attr_mask |= C.UCP_OP_ATTR_FIELD_CALLBACK | C.UCP_OP_ATTR_FIELD_USER_DATA
			cbAddr := (*C.ucp_am_recv_data_nbx_callback_t)(unsafe.Pointer(&requestParams.cb[0]))
			*cbAddr = (C.ucp_am_recv_data_nbx_callback_t)(C.ucxgo_completeAmRecvData)

			requestParams.user_data = unsafe.Pointer(uintptr(cbId))
		}
	}

	request := C.ucp_am_recv_data_nbx(w.worker, dataDesc.dataPtr, recvBuffer, C.size_t(size), &requestParams)

	return NewRequest(request, cbId, length)
}
