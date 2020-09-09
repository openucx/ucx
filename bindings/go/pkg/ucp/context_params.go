package ucp

/*
#include <ucp/api/ucp.h>
*/
import "C"

type contextParams struct {
	params C.ucp_params_t
	config map[string]string
}

func NewContextParams() *contextParams {
	return &contextParams{params: C.ucp_params_t{}, config: map[string]string{}}
}

func (p *contextParams) SetTagSenderMask(tagSenderMask uint64) {
	p.params.tag_sender_mask = C.ulong(tagSenderMask)
	p.params.field_mask |= C.UCP_PARAM_FIELD_TAG_SENDER_MASK
}

func (p *contextParams) SetEstimatedNumEPS(estimatedNumEPS uint32) {
	p.params.estimated_num_eps = C.ulong(estimatedNumEPS)
	p.params.field_mask |= C.UCP_PARAM_FIELD_ESTIMATED_NUM_EPS
}

func (p *contextParams) EnableSharedWorkers() {
	p.params.mt_workers_shared = 1
	p.params.field_mask |= C.UCP_PARAM_FIELD_MT_WORKERS_SHARED
}

func (p *contextParams) EnableTag() {
	p.params.features |= C.UCP_FEATURE_TAG
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
}

func (p *contextParams) EnableRMA() {
	p.params.features |= C.UCP_FEATURE_RMA
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
}

func (p *contextParams) EnableAtomic32Bit() {
	p.params.features |= C.UCP_FEATURE_AMO32
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
}

func (p *contextParams) EnableAtomic64Bit() {
	p.params.features |= C.UCP_FEATURE_AMO64
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
}

func (p *contextParams) EnableWakeup() {
	p.params.features |= C.UCP_FEATURE_WAKEUP
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
}

func (p *contextParams) EnableStream() {
	p.params.features |= C.UCP_FEATURE_STREAM
	p.params.field_mask |= C.UCP_PARAM_FIELD_FEATURES
}

func (p *contextParams) SetConfig(key, value string) {
	p.config[key] = value
}
