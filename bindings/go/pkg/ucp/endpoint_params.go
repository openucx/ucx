package ucp

/*
#include <string.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include <netdb.h>

typedef struct sockaddr sockaddr_t;

int set_sock_storage_ip(char *addr, uint16_t port, sockaddr_t *sa, unsigned int *len) {
	int err;
	struct addrinfo *addrinfo;

	err = getaddrinfo(addr, NULL, NULL, &addrinfo);
	if (err != 0) {
		return err;
	}

	if (addrinfo->ai_family == AF_INET6) {
		memcpy(sa, &addrinfo->ai_addr, sizeof(struct sockaddr_in6));
		((struct sockaddr_in6*)sa)->sin6_port = port;
		*len = sizeof(struct sockaddr_in6);
	} else {
		memcpy(sa, &addrinfo->ai_addr, sizeof(struct sockaddr_in));
		((struct sockaddr_in*)sa)->sin_port = port;
		*len = sizeof(struct sockaddr_in);
	}

	freeaddrinfo(addrinfo);
	return 0;
}

void endpointErrorHandlerProxy(void *arg, ucp_ep_h ep, ucs_status_t status);

void error_handler_callback(void *arg, ucp_ep_h ep, ucs_status_t status) {
	endpointErrorHandlerProxy(arg, ep, status);
}
*/
import "C"
import (
	"errors"
	"unsafe"
)

type endpointParams struct {
	params C.ucp_ep_params_t
}

type errorHandlerArg struct {
	cb  func(interface{})
	arg interface{}
}

func NewEndpointParams() *endpointParams {
	return &endpointParams{params: C.ucp_ep_params_t{}}
}

func (p *endpointParams) SetRemoteAddress(ucpAddress *C.ucp_address_t) {
	p.params.address = ucpAddress
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_REMOTE_ADDRESS
}

func (p *endpointParams) EnablePeerErrorHandlingMode() {
	p.params.err_mode = C.UCP_ERR_HANDLING_MODE_PEER
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE
}

func (p *endpointParams) SetSocketAddressIP(ip string, port int) error {
	var sa C.sockaddr_t
	var saLen C.uint

	if err := C.set_sock_storage_ip(C.CString(ip), C.ushort(port), &sa, &saLen); err != 0 {
		return errors.New("Unable to set IP")
	}

	p.params.flags |= C.UCP_EP_PARAMS_FLAGS_CLIENT_SERVER
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_SOCK_ADDR | C.UCP_EP_PARAM_FIELD_FLAGS
	p.params.sockaddr.addr = &sa
	p.params.sockaddr.addrlen = saLen
	return nil
}

func (p *endpointParams) EnableNoLoopbackMode() {
	p.params.flags |= C.UCP_EP_PARAMS_FLAGS_NO_LOOPBACK
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_FLAGS
}

func (p *endpointParams) SetConnectionRequest(connectionRequest C.ucp_conn_request_h) {
	p.params.conn_request = connectionRequest
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_CONN_REQUEST
}

func (p *endpointParams) SetErrorHandler(f func(interface{}), data interface{}) {
	var eh C.ucp_err_handler_t
	eh.cb = C.ucp_err_handler_cb_t(C.error_handler_callback)
	eh.arg = unsafe.Pointer(&errorHandlerArg{f, data})
	p.params.err_handler = eh
	p.params.field_mask |= C.UCP_EP_PARAM_FIELD_ERR_HANDLER
}
