package ucp

/*
#include <ucp/api/ucp.h>
*/
import "C"
import "fmt"

type workerContext struct {
	ctx       C.ucp_context_h
	workerCtx C.ucp_worker_h
	params    C.ucp_worker_params_t
}

func newWorker(ctx *context, params *workerParams) (*workerContext, error) {
	var ucp_worker C.ucp_worker_h

	if status := C.ucp_worker_create(ctx.ctx, &params.params, &ucp_worker); status != C.UCS_OK {
		return nil, fmt.Errorf("Unable to create worker: %d", status)
	}

	workerCtx := &workerContext{
		ctx:       ctx.ctx,
		workerCtx: ucp_worker,
		params:    params.params}
	return workerCtx, nil
}
