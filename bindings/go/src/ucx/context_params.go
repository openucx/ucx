/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
import "C"
import (
	"runtime"
	"unsafe"
)

// Tuning parameters for UCP library.
// The structure defines the parameters that are used for
// UCP library tuning during UCP library "initialization".
//
// UCP library implementation uses the "features"
// parameter to optimize the library functionality that minimize memory
// footprint. For example, if the application does not require send/receive
// semantics UCP library may avoid allocation of expensive resources associated with
// send/receive queues.
type UcpParams struct {
	params C.ucp_params_t
}

// Mask which specifies particular bits of the tag which can uniquely
// identify the sender (UCP endpoint) in tagged operations.
// This field defaults to 0 if not specified.
func (p *UcpParams) SetTagSenderMask(tagSenderMask uint64) *UcpParams {
	p.params.tag_sender_mask = C.ulong(tagSenderMask)
	p.params.field_mask |= C.UCP_PARAM_FIELD_TAG_SENDER_MASK
	return p
}

// An optimization hint of how many endpoints would be created on this context.
// Does not affect semantics, but only transport selection criteria and the
// resulting performance.
// The value can be also set by UCX_NUM_EPS environment variable. In such case
// it will override the number of endpoints set by this method.
func (p *UcpParams) SetEstimatedNumEPS(estimatedNumEPS uint64) *UcpParams {
	p.params.estimated_num_eps = C.ulong(estimatedNumEPS)
	p.params.field_mask |= C.UCP_PARAM_FIELD_ESTIMATED_NUM_EPS
	return p
}

// An optimization hint for a single node. For example, when used from MPI or
// OpenSHMEM libraries, this number will specify the number of Processes Per
// Node (PPN) in the job. Does not affect semantics, only transport selection
// criteria and the resulting performance.
// The value can be also set by the UCX_NUM_PPN environment variable, which
// will override the number of endpoints set by this method.
func (p *UcpParams) SetEstimatedNumPPN(estimatedNumPPN uint64) *UcpParams {
	p.params.estimated_num_ppn = C.ulong(estimatedNumPPN)
	p.params.field_mask |= C.UCP_PARAM_FIELD_ESTIMATED_NUM_PPN
	return p
}

// Tracing and analysis tools can identify the context using this name.
func (p *UcpParams) SetName(name string) *UcpParams {
	freeParamsName(p)
	p.params.name = C.CString(name)
	runtime.SetFinalizer(p, func(f *UcpParams) { FreeNativeMemory(unsafe.Pointer(f.params.name)) })
	p.params.field_mask |= C.UCP_PARAM_FIELD_NAME
	return p
}

// Indicates if this context is shared by multiple workers
// from different threads. If so, this context needs thread safety
// support; otherwise, the context does not need to provide thread
// safety.
// For example, if the context is used by single worker, and that
// worker is shared by multiple threads, this context does not need
// thread safety; if the context is used by worker 1 and worker 2,
// and worker 1 is used by thread 1 and worker 2 is used by thread 2,
// then this context needs thread safety.
// Note that actual thread mode may be different from mode passed
// to UcpContext.
func (p *UcpParams) EnableSharedWorkers() *UcpParams {
	p.params.mt_workers_shared = 1
	p.params.field_mask |= C.UCP_PARAM_FIELD_MT_WORKERS_SHARED
	return p
}

// Request tag matching support.
func (p *UcpParams) EnableTag() *UcpParams {
	p.params.features |= C.UCP_FEATURE_TAG
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
	return p
}

// Request remote memory access support.
func (p *UcpParams) EnableRMA() *UcpParams {
	p.params.features |= C.UCP_FEATURE_RMA
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
	return p
}

// Request 32-bit atomic operations support.
func (p *UcpParams) EnableAtomic32Bit() *UcpParams {
	p.params.features |= C.UCP_FEATURE_AMO32
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
	return p
}

// Request 64-bit atomic operations support.
func (p *UcpParams) EnableAtomic64Bit() *UcpParams {
	p.params.features |= C.UCP_FEATURE_AMO64
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
	return p
}

// Request interrupt notification support.
func (p *UcpParams) EnableWakeup() *UcpParams {
	p.params.features |= C.UCP_FEATURE_WAKEUP
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
	return p
}

// Request stream support.
func (p *UcpParams) EnableStream() *UcpParams {
	p.params.features |= C.UCP_FEATURE_STREAM
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
	return p
}

// Request Active Message support feature.
func (p *UcpParams) EnableAM() *UcpParams {
	p.params.features |= C.UCP_FEATURE_AM
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
	return p
}
