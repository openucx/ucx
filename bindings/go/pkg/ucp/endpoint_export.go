package ucp

/*
#include <ucp/api/ucp.h>
*/
import "C"

import "unsafe"

//export endpointErrorHandlerProxy
func endpointErrorHandlerProxy(arg unsafe.Pointer, ep C.ucp_ep_h, status C.ucs_status_t) {
	eh := (*errorHandlerArg)(arg)
	eh.cb(eh.arg)
}
