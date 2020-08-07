package ucp

/*
#include <ucp/api/ucp.h>
#include <ucs/config/global_opts.h>
#include <ucs/type/spinlock.h>

struct go_ucp_context {
    ucs_status_t status;
    ucs_recursive_spinlock_t lock;
	size_t length;
};

const size_t GO_UCP_CONTEXT_SIZE = sizeof(struct go_ucp_context);

void go_ucp_request_init(void *request) {
     struct go_ucp_context *ctx = (struct go_ucp_context *)request;
     ctx->status = UCS_INPROGRESS;
     ctx->length = 0;
     ucs_recursive_spinlock_init(&ctx->lock, 0);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"ucx/pkg/ucs"
)

type context struct {
	ctx    C.ucp_context_h
	params C.ucp_params_t
}

func NewContext(p *contextParams) (*context, error) {
	p.params.request_size = C.GO_UCP_CONTEXT_SIZE
	p.params.request_init = C.ucp_request_init_callback_t(C.go_ucp_request_init)
	p.params.field_mask |= C.UCP_PARAM_FIELD_REQUEST_INIT | C.UCP_PARAM_FIELD_REQUEST_SIZE

	ucp_config := &C.ucp_config_t{}
	if status := C.ucp_config_read(nil, nil, &ucp_config); status != C.UCS_OK {
		ucs.LogErrorf("Unable to read config: %d", status)
		return nil, fmt.Errorf("Unable to read config: %d", status)
	}

	for k, v := range p.config {
		k_cstr := C.CString(k)
		v_cstr := C.CString(v)

		configStatus := C.ucp_config_modify(ucp_config, k_cstr, v_cstr)
		globalStatus := C.ucs_global_opts_set_value(k_cstr, v_cstr)
		if configStatus != C.UCS_OK && globalStatus != C.UCS_OK {
			ucs.LogWarnf("No such key %s, ignoring", k)
		}

		C.free(unsafe.Pointer(v_cstr))
		C.free(unsafe.Pointer(k_cstr))
	}

	var ucp_context C.ucp_context_h
	if status := C.ucp_init(&p.params, ucp_config, &ucp_context); status != C.UCS_OK {
		ucs.LogErrorf("No such key %s, ignoring", status)
		return nil, fmt.Errorf("Unable to init config: %d", status)
	}

	if ucp_config != nil {
		C.ucp_config_release(ucp_config)
	}

	ctx := &context{
		ctx:    ucp_context,
		params: p.params}
	return ctx, nil
}

func (c *context) NewWorker(params *workerParams) (*workerContext, error) {
	return newWorker(c, params)
}

func (c *context) Close() error {
	C.ucp_cleanup(c.ctx)
	return nil
}
