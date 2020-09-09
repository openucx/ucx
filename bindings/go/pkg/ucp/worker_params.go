package ucp

/*
#include <ucp/api/ucp.h>

void go_ucs_cpu_set(int index, ucs_cpu_set_t *cpu_mask) {
	UCS_CPU_SET(index, cpu_mask);
}
*/
import "C"
import (
	"strconv"
	"unsafe"
)

type workerParams struct {
	params C.ucp_worker_params_t
}

func NewWorkerParams() *workerParams {
	params := C.ucp_worker_params_t{}
	return &workerParams{params: params}
}

func (p *workerParams) EnableMultiThread() {
	p.params.thread_mode = C.UCS_THREAD_MODE_MULTI
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_THREAD_MODE
}

func (p *workerParams) SetCPUMask(cpuMask uint64) {
	var cpu_mask C.ucs_cpu_set_t
	bits := strconv.FormatUint(cpuMask, 2)
	n := len(bits)
	for i, b := range bits {
		if b == '1' {
			C.go_ucs_cpu_set(C.int(n-i-1), &cpu_mask)
		}
	}
	p.params.cpu_mask = cpu_mask
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_CPU_MASK
}

func (p *workerParams) EnableWakeupRMAEvent() {
	p.params.events |= C.UCP_WAKEUP_RMA
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
}

func (p *workerParams) EnableWakeupAMOEvent() {
	p.params.events |= C.UCP_WAKEUP_AMO
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
}

func (p *workerParams) EnableWakeupTagSendEvent() {
	p.params.events |= C.UCP_WAKEUP_TAG_SEND
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
}

func (p *workerParams) EnableWakeupTagRecvEvent() {
	p.params.events |= C.UCP_WAKEUP_TAG_RECV
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
}

func (p *workerParams) EnableWakeupTXEvent() {
	p.params.events |= C.UCP_WAKEUP_TX
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
}

func (p *workerParams) EnableWakeupRXEvent() {
	p.params.events |= C.UCP_WAKEUP_RX
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
}

func (p *workerParams) EnableWakeupEdgeEvent() {
	p.params.events |= C.UCP_WAKEUP_EDGE
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENTS
}

func (p *workerParams) SetUserData(userData unsafe.Pointer) {
	p.params.user_data = userData
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_USER_DATA
}

func (p *workerParams) SetEventFD(eventFD int) {
	p.params.event_fd = C.int(eventFD)
	p.params.field_mask |= C.UCP_WORKER_PARAM_FIELD_EVENT_FD
}
