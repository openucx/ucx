package http

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/Artemy-Mellanox/ucx/bindings/go/src/ucx"
)

const AM_REQ = 1
const AM_RESP = 2

func getBuf(buf []byte) (unsafe.Pointer, uint64) {
	if len(buf) > 0 {
		return unsafe.Pointer(&buf[0]), uint64(len(buf))
	}
	return nil, 0
}

type context struct {
	context *ucx.UcpContext
	worker *ucx.UcpWorker
}

func (c *context) Init() {
	contextParams := ucx.UcpParams{}
	contextParams.EnableAM()
	context, err := ucx.NewUcpContext(&contextParams)
	if err != nil {
		log.Fatalf("Failed to create UCX context: %v", err)
	}

	workerParams := ucx.UcpWorkerParams{}
	workerParams.SetThreadMode(ucx.UCS_THREAD_MODE_MULTI)
	worker, err := context.NewWorker(&workerParams)
	if err != nil {
		log.Fatalf("Failed to create UCX worker: %v", err)
	}

	c.context = context
	c.worker = worker
}

func (c *context) Close() {
	c.worker.Close()
	c.context.Close()
}

type Server struct {
	context
	listener *ucx.UcpListener
	handler http.Handler

	eps []*ucx.UcpEp
	reqs []*ucx.UcpRequest
}

type responseWriter struct {
	server   *Server
	replyEp  *ucx.UcpEp
	headers  http.Header
	status   int
	body     []byte
	bodyLock sync.Mutex
	done     chan bool
	content  bytes.Buffer
	left     int
	id	 string
}

func (w *responseWriter) Header() http.Header {
	return w.headers
}

func (w *responseWriter) onData(request *ucx.UcpRequest, status ucx.UcsStatus) {
	request.Close()
	w.done <- true
}

func (w *responseWriter) Write(data []byte) (int, error) {
	w.bodyLock.Lock()
	defer w.bodyLock.Unlock()

	var body []byte
	if w.left == 0 {
		w.left, _ = strconv.Atoi(w.headers.Get("Content-Length"))
		if w.left == 0 {
			w.left = len(data)
			w.headers.Set("Content-Length", strconv.Itoa(w.left))
		}
		if w.left == len(data) {
			body = data
		}
	}

	if body == nil {
		w.content.Write(data)
		w.left -= len(data)
		if w.left > 0 {
			return len(data), nil
		}
		body = w.content.Bytes()
	}

	headerMap := map[string]string{
		"ucx-code": strconv.Itoa(w.status),
		"ucx-id": w.id,
	}

	for k, v := range w.headers {
		headerMap[k] = v[0]
	}

	header, err := json.Marshal(headerMap)
	if err != nil {
		return 0, err
	}

	w.done = make(chan bool, 1)
	reqParams := &ucx.UcpRequestParams{}
	reqParams.SetCallback(w.onData)

	headerPtr, headerLen := getBuf(header)
	dataPtr, dataLen := getBuf(body)
	_, err = w.replyEp.SendAmNonBlocking(AM_RESP,
		headerPtr, headerLen, dataPtr, dataLen,
		0, reqParams)
	if err != nil {
		return 0, err
	}

	<-w.done
	return len(data), nil
}

func (w *responseWriter) WriteHeader(statusCode int) {
	w.status = statusCode
	w.Write(nil)
}

type amDataReader struct {
	id string
	data *ucx.UcpAmData
	left int
	ptr unsafe.Pointer
	b []byte
	done chan bool
	bodyLock sync.Mutex
	needClose bool
}

func (r *amDataReader) onData(request *ucx.UcpRequest, status ucx.UcsStatus, length uint64) {
	request.Close()
	r.done <- true
}

func (r *amDataReader) Read(p []byte) (int, error) {
	if r.left == 0 {
		return 0, io.EOF
	}
	doCopy := true
	if r.ptr == nil {
		if r.data.IsDataValid() {
			r.ptr, _ = r.data.DataPointer()
			r.needClose = true
		} else {
			var s uint64
			r.done = make(chan bool, 1)
			reqParams := &ucx.UcpRequestParams{}
			reqParams.SetCallback(r.onData)
			if len(p) >= int(r.data.Length()) {
				r.ptr, s = getBuf(p)
				doCopy = false
			} else {
				r.b = make([]byte, r.data.Length())
				r.ptr, s = getBuf(r.b)
			}
			_, err := r.data.Receive(r.ptr, s, reqParams)
			if err != nil {
				return 0, err
			}
			<-r.done
		}
	}

	var res error
	l := r.left
	if l > len(p) {
		l = len(p)
	}
	if doCopy {
		C.memcpy(unsafe.Pointer(&p[0]), r.ptr, C.size_t(l))
	}
	r.left -= l
	if r.left == 0 {
		res = io.EOF
	} else {
		r.ptr = unsafe.Pointer(uintptr(r.ptr) + uintptr(l))
	}
	return int(l), res
}

func (r *amDataReader) Close() (error) {
	if r.needClose {
		r.data.Close()
	}
	return nil
}

func handleAmResp(header unsafe.Pointer, headerSize uint64, data *ucx.UcpAmData) (map[string]string, *amDataReader, error) {
	var headerMap map[string]string
	if headerSize > 0 {
		headerBytes := ucx.GoBytes(header, headerSize)
		if err := json.Unmarshal(headerBytes, &headerMap); err != nil {
			return nil, nil, err
		}
	}

	r := &amDataReader{
		data: data,
		left: int(data.Length()),
	}

	if data.Length() == 0 {
		r.needClose = true
	}

	if !data.CanPersist() {
		l := data.Length()
		if (l > 0) {
			ptr, err := data.DataPointer();
			if err != nil {
				fmt.Printf("handleAmResp %v\n", err)
				return nil, nil, err
			}

			r.b = make([]byte, l)
			r.ptr = unsafe.Pointer(&r.b[0])
			C.memcpy(r.ptr, ptr, C.size_t(l))
		}
	}

	return headerMap, r, nil
}

func (s *Server) response(header unsafe.Pointer, headerSize uint64, data *ucx.UcpAmData, replyEp *ucx.UcpEp) ucx.UcsStatus {
	headerMap, reader, err := handleAmResp(header, headerSize, data)
	if err != nil {
		fmt.Printf("response %v\n", err)
		return ucx.UCS_ERR_IO_ERROR
	}
	req, _ := http.NewRequest(headerMap["ucx-method"], headerMap["ucx-url"], reader)
	for k, v := range headerMap {
		if !strings.HasPrefix(k, "ucx-") {
			req.Header.Set(k,v)
		}
	}
	req.ContentLength = int64(data.Length())
	req.RequestURI = req.URL.EscapedPath()
	writer := &responseWriter{
		server: s,
		replyEp: replyEp,
		headers: make(http.Header),
		id: headerMap["ucx-id"],
	}
	s.eps = append(s.eps, replyEp)
	reader.id = headerMap["ucx-id"]

	go s.handler.ServeHTTP(writer, req)
	if data.CanPersist() {
		return ucx.UCS_INPROGRESS
	} else {
		return ucx.UCS_OK
	}
}

func onErr(ep *ucx.UcpEp, status ucx.UcsStatus) {
	if status == ucx.UCS_ERR_CONNECTION_RESET {
		return
	}
	errorString := fmt.Sprintf("Endpoint error: %v", status.String())
	panic(errorString)
}

func (s *Server) servConn(conn *ucx.UcpConnectionRequest) {
	epParams := &ucx.UcpEpParams{}
	epParams.SetConnRequest(conn)
	epParams.SetErrorHandler(onErr)
	_, err := s.worker.NewEndpoint(epParams)
	if err != nil {
		fmt.Printf("Failed to create endpoint: %v\n", err)
		return
	}
}

func (s *Server) Close() {
	s.listener.Close()
	s.context.Close()
}

func NewServer(addr string, handler http.Handler) (*Server, error) {
	s := &Server{
		handler: handler,
	}
	s.Init()

	s.worker.SetAmRecvHandler(AM_REQ, ucx.UCP_AM_FLAG_PERSISTENT_DATA, s.response)

	tcp, err := net.ResolveTCPAddr("tcp", addr)
	if err != nil {
		return nil, err
	}

	listenerParams := &ucx.UcpListenerParams{}
	listenerParams.SetSocketAddress(tcp)
	listenerParams.SetConnectionHandler(s.servConn)
	listener, err := s.worker.NewListener(listenerParams);
	if err != nil {
		return nil, err
	}

	s.listener = listener
	return s, nil
}

func (s *Server) Serve() {
	for {
		//if n := s.worker.Progress(); n > 0 { fmt.Printf("Server Progress %d\n", n) }
		s.worker.Progress();
	}
}

func StartServer(addr string, handler http.Handler) (serve func() error, err error) {
	s, err := NewServer(addr, handler)
	if (err != nil) {
		return nil, err
	}
	serve = func() error {
		s.Serve()
		return nil
	}
	return
}

type tracker struct {
	resp chan *http.Response
	id string
}

type Transport struct {
	context
	ep *ucx.UcpEp
	currId int32
	reqs sync.Map
	quit chan bool
}

func (a *Transport) recvResponse(header unsafe.Pointer, headerSize uint64, data *ucx.UcpAmData, replyEp *ucx.UcpEp) ucx.UcsStatus {
	headerMap, reader, err := handleAmResp(header, headerSize, data)
	if err != nil {
		fmt.Printf("recvResponse %v\n", err)
		return ucx.UCS_ERR_IO_ERROR
	}
	resp := &http.Response{
		Header: make(http.Header),
		Body: reader,
		Status: headerMap["ucx-code"],
	}

	statusCode, _, _ := strings.Cut(resp.Status, " ")
	resp.StatusCode, _ = strconv.Atoi(statusCode)
	contentLength, _ := strconv.Atoi(headerMap["Content-Length"])
	resp.ContentLength = int64(contentLength)

	for k, v := range headerMap {
		resp.Header.Set(k,v)
	}

	req, _ := a.reqs.Load(headerMap["ucx-id"])
	req.(*tracker).resp <- resp
	if data.CanPersist() {
		return ucx.UCS_INPROGRESS
	} else {
		return ucx.UCS_OK
	}
}

func (a *Transport) Close() {
	a.quit <- true
	a.ep.CloseNonBlockingForce(nil)
	a.context.Close()
}

func (a *Transport) progress() {
	for {
		select {
			case <-a.quit: return
			default: a.worker.Progress()
			//if n := a.worker.Progress(); n > 0 { fmt.Printf("Transport Progress %d\n", n) }
		}
	}
}

func NewTransport(addr string) (*Transport, error) {
	a := new(Transport)
	a.Init()
	a.quit = make(chan bool, 1)

	a.worker.SetAmRecvHandler(AM_RESP, ucx.UCP_AM_FLAG_PERSISTENT_DATA, a.recvResponse)

	tcp, err := net.ResolveTCPAddr("tcp", addr)
	if err != nil {
		return nil, err
	}

	epParams := &ucx.UcpEpParams{}
	epParams.SetSocketAddress(tcp)
	epParams.SetErrorHandler(onErr)
	ep, err := a.worker.NewEndpoint(epParams)
	if err != nil {
		return nil, err
	}

	a.ep = ep
	go a.progress()
	return a, nil
}

func (a *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	reqId := strconv.Itoa(int(atomic.AddInt32(&a.currId, 1)))
	headerMap := map[string]string{
		"ucx-method": req.Method,
		"ucx-url": req.URL.String(),
		"ucx-id": reqId,
	}

	for k, v := range req.Header {
		headerMap[k] = v[0]
	}

	header, err := json.Marshal(headerMap)
	if err != nil {
		return nil, err
	}

	var data []byte
	if req.Body != nil {
		data, err = ioutil.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
	}

	tr := &tracker{ resp: make(chan *http.Response, 1), id: reqId }
	a.reqs.Store(reqId, tr)

	headerPtr, headerLen := getBuf(header)
	dataPtr, dataLen := getBuf(data)
	send, err := a.ep.SendAmNonBlocking(AM_REQ,
		headerPtr, headerLen, dataPtr, dataLen,
		ucx.UCP_AM_SEND_FLAG_REPLY, nil)
	if err != nil {
		return nil, err
	}
	defer send.Close()
	resp := <-tr.resp
	a.reqs.Delete(reqId)

	return resp, nil
}
