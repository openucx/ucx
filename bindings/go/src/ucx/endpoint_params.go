/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucx

// #include <ucp/api/ucp.h>
// #include "goucx.h"
import "C"

// Tuning parameters for the UCP endpoint.
type UcpEpParams struct {
	params       C.ucp_ep_params_t
	errorHandler UcpEpErrHandler
}

// This callback routine is invoked when transport level error detected.
// ep - Endpoint to handle transport level error. Upon return
// from the callback, this endpoint is no longer usable and
// all subsequent operations on this ep will fail with
// the error code passed in status.
type UcpEpErrHandler func(ep *UcpEp, status UcsStatus)

//export ucxgo_completeGoErrorHandler
func ucxgo_completeGoErrorHandler(ep C.ucp_ep_h, status C.ucs_status_t) {
	errHandleGoCallback := errorHandles[ep]
	errHandleGoCallback(&UcpEp{
		ep: ep,
	}, UcsStatus(status))
}

// Destination address
func (p *UcpEpParams) SetUcpAddress(a *UcpAddress) *UcpEpParams {
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_REMOTE_ADDRESS
	p.params.address = a.Address
	return p
}

// Guarantees that send requests are always completed (successfully or error) even in
// case of remote failure, disables protocols and APIs which may cause a hang or undefined
// behavior in case of peer failure, may affect performance and memory footprint
func (p *UcpEpParams) SetPeerErrorHandling() *UcpEpParams {
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE
	p.params.err_mode = C.ucp_err_handling_mode_t(C.UCP_ERR_HANDLING_MODE_PEER)
	return p
}

// Handler to process transport level failure.
func (p *UcpEpParams) SetErrorHandler(errHandler UcpEpErrHandler) *UcpEpParams {
	var err_handler_t C.ucp_err_handler_t
	err_handler_t.cb = (C.ucp_err_handler_cb_t)(C.ucxgo_completeGoErrorHandler)
	p.errorHandler = errHandler
	p.params.err_handler = err_handler_t
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_ERR_HANDLER
	return p
}

// Tracing and analysis tools can identify the endpoint using this name.
func (p *UcpEpParams) SetName(name string) *UcpEpParams {
	p.params.name = C.CString(name)
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_NAME
	return p
}
