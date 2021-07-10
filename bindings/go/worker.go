/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucp

// #include <ucp/api/ucp.h>
import "C"

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

func (m *UcpWorker) Close() {
	C.ucp_worker_destroy(m.worker)
}

func (m *UcpWorker) Query(attrs ...UcpWorkerAttribute) (*C.ucp_worker_attr_t, error) {
	var workerAttr C.ucp_worker_attr_t

	for attr, _ := range attrs {
		workerAttr.field_mask |= C.ulong(attr)
	}

	if status := C.ucp_worker_query(m.worker, &workerAttr); status != C.UCS_OK {
		return nil, NewUcxError(status)
	}

	return &workerAttr, nil
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
func (m *UcpWorker) Arm() UcsStatus {
	return UcsStatus(C.ucp_worker_arm(m.worker))
}

// This routine explicitly progresses all communication operations on a worker.
// Typically, request wait and test routines call UcpWorker.Progress()
// "this routine" to progress any outstanding operations.
// Transport layers, implementing asynchronous progress using threads,
// require callbacks and other user code to be thread safe.
// The state of communication can be advanced (progressed) by blocking
// routines. Nevertheless, the non-blocking routines can not be used for
// communication progress.
func (m *UcpWorker) Progress() uint {
	return uint(C.ucp_worker_progress(m.worker))
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
func (m *UcpWorker) Wait() error {
	if status := C.ucp_worker_wait(m.worker); status != C.UCS_OK {
		return NewUcxError(status)
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
func (m *UcpWorker) GetEfd() (int, error) {
	var efd C.int
	if status := C.ucp_worker_get_efd(m.worker, &efd); status != C.UCS_OK {
		return 0, NewUcxError(status)
	}
	return int(efd), nil
}

// This routine signals that the event has happened, as part of the wake-up
// mechanism. This function causes a blocking call to UcpWorker.Wait() or
// waiting on a file descriptor from UcpWorker.GetEfd() to return, even
// if no event from the underlying interfaces has taken place.
func (m *UcpWorker) Signal() error {
	if status := C.ucp_worker_signal(m.worker); status != C.UCS_OK {
		return NewUcxError(status)
	}
	return nil
}
