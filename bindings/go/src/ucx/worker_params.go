/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
// void cpu_zero(ucs_cpu_set_t *cpu_mask) {
//   UCS_CPU_ZERO(cpu_mask);
// }
// void set_cpu(int cpuId, ucs_cpu_set_t *cpu_mask) {
// 	 UCS_CPU_SET(cpuId, cpu_mask);
// }
import "C"
import (
	"math/big"
	"runtime"
	"unsafe"
)

// Tuning parameters for the UCP worker.
type UcpWorkerParams struct {
	params C.ucp_worker_params_t
}

// The parameter thread_mode suggests the thread safety mode which worker
// and the associated resources should be created with. This is an
// optional parameter. The default value is UCS_THREAD_MODE_SINGLE and
// it is used when the value of the parameter is not set. When this
// parameter is set, the UcpContext.NewWorker() attempts to create worker with this thread mode.
// The thread mode with which worker is created can differ from the
// suggested mode. The actual thread mode of the worker should be obtained
// using the query interface UcpWorker.Query().
func (p *UcpWorkerParams) SetThreadMode(threadMode UcsThreadMode) *UcpWorkerParams {
	p.params.thread_mode = C.ucs_thread_mode_t(threadMode)
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_THREAD_MODE
	return p
}

// Mask of which CPUs worker resources should preferably be allocated on.
// This value is optional. If it's not set, resources are allocated according to system's default policy.
func (p *UcpWorkerParams) SetCpuMask(mask *big.Int) *UcpWorkerParams {
	var cpu_mask C.ucs_cpu_set_t
	C.cpu_zero(&cpu_mask)
	for i := 0; i < mask.BitLen(); i++ {
		if mask.Bit(i) != 0 {
			C.set_cpu(C.int(i), &cpu_mask)
		}
	}
	p.params.cpu_mask = cpu_mask
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_CPU_MASK
	return p
}

// Mask of events (UcpWakeupEvent) which are expected on wakeup.
// This value is optional.
// If it's not set, all types of events will trigger on wakeup.
func (p *UcpWorkerParams) SetWakeupEvent(event UcpWakeupEvent) *UcpWorkerParams {
	if p.params.field_mask&C.UCP_WORKER_PARAM_FIELD_EVENTS == 0 {
		p.params.events = 0
	}
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
	p.params.events |= C.uint(event)
	return p
}

// Wakeup on remote memory access send completion.
func (p *UcpWorkerParams) WakeupRMA() *UcpWorkerParams {
	return p.SetWakeupEvent(UCP_WAKEUP_RMA)
}

// Wakeup on atomic operation send completion.
func (p *UcpWorkerParams) WakeupAMO() *UcpWorkerParams {
	return p.SetWakeupEvent(UCP_WAKEUP_AMO)
}

// Wakeup on tag send completion.
func (p *UcpWorkerParams) WakeupTagSend() *UcpWorkerParams {
	return p.SetWakeupEvent(UCP_WAKEUP_TAG_SEND)
}

// Wakeup on tag recv completion.
func (p *UcpWorkerParams) WakeupTagRecv() *UcpWorkerParams {
	return p.SetWakeupEvent(UCP_WAKEUP_TAG_RECV)
}

// This event type will generate an event on completion of any
// outgoing operation (complete or  partial, according to the
// underlying protocol) for any type of transfer (send, atomic, or RMA).
func (p *UcpWorkerParams) WakeupTX() *UcpWorkerParams {
	return p.SetWakeupEvent(UCP_WAKEUP_TX)
}

// This event type will generate an event on completion of any receive
// operation (complete or partial, according to the underlying protocol).
func (p *UcpWorkerParams) WakeupRX() *UcpWorkerParams {
	return p.SetWakeupEvent(UCP_WAKEUP_RX)
}

// Use edge-triggered wakeup. The event file descriptor will be signaled only
// for new events, rather than existing ones.
func (p *UcpWorkerParams) WakeupEdge() *UcpWorkerParams {
	return p.SetWakeupEvent(UCP_WAKEUP_EDGE)
}

// User data associated with the current worker.
func (p *UcpWorkerParams) SetUserData(data []byte) *UcpWorkerParams {
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_USER_DATA
	p.params.user_data = C.CBytes(data)
	return p
}

// External event file descriptor.
// Events on the worker will be reported on the provided event file descriptor.
// The provided file descriptor must be capable of aggregating notifications
// for arbitrary events, for example epoll(7) on Linux systems.
// userData will be used as the event user-data on systems which
// support it. For example, on Linux, it will be placed in
// epoll_data_t::ptr, when returned from epoll_wait(2).
// Otherwise, events would be reported to the event file descriptor returned
// from UcpWorker.GetEfd().
func (p *UcpWorkerParams) SetEventFD(fd uintptr) *UcpWorkerParams {
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENT_FD
	p.params.event_fd = C.int(fd)
	return p
}

// Tracing and analysis tools can identify the worker using this name.
func (p *UcpWorkerParams) SetName(name string) *UcpWorkerParams {
	freeParamsName(p)
	p.params.name = C.CString(name)
	runtime.SetFinalizer(p, func(f *UcpWorkerParams) { FreeNativeMemory(unsafe.Pointer(f.params.name)) })
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_NAME
	return p
}

// Minimal address alignment of the active message data pointer as passed
// in argument data to the active message handle
func (p *UcpWorkerParams) SetAmAlignment(alignment uint64) *UcpWorkerParams {
	p.params.am_alignment = C.size_t(alignment)
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_AM_ALIGNMENT
	return p
}

// Client id that is sent as part of the connection request payload when connecting to a remote socket address.
// On the remote side, this value can be obtained by calling
// UcpConnectionRequest.Query(UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID)
func (p *UcpWorkerParams) SetClientId(clientId uint64) *UcpWorkerParams {
	p.params.client_id = C.uint64_t(clientId)
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_CLIENT_ID
	return p
}
