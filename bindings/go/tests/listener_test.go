/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package goucxtests

import (
	"net"
	"testing"
	. "ucx"
)

var connReq *UcpConnectionRequest = nil
var rejected bool = false

func TestUcpListener(t *testing.T) {
	const clientId = 1
	addr, _ := net.ResolveTCPAddr("tcp", "0.0.0.0")

	ucpParams := (&UcpParams{}).EnableTag()

	ucpWorkerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_SINGLE)

	listenerParams := &UcpListenerParams{}
	listenerParams.SetSocketAddress(addr)

	epParams := &UcpEpParams{}
	var logErrorHandler UcpEpErrHandler = func(ep *UcpEp, status UcsStatus) {
		t.Fatalf("Error handler called with status %d", status)
	}
	epParams.SetPeerErrorHandling().SetErrorHandler(logErrorHandler).SendClientId()

	listenerParams.SetConnectionHandler(func(connRequest *UcpConnectionRequest) {
		connRequestParams, err := connRequest.Query(UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID, UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR)
		if err != nil {
			t.Fatalf("Failed to query connection request %v", err)
		}

		if connRequestParams.ClientId != clientId {
			t.Fatalf("Client id %v != %v", connRequestParams.ClientId, clientId)
		}

		if connRequestParams.ClientAddress == nil {
			t.Fatalf("Client address is empty")
		}
		// Use first request to create backward ep with epParams.SetConnRequest()
		// And second request reject to test ep error handling.
		if connReq == nil {
			connReq = connRequest
		} else {
			connRequest.Reject()
		}
	})

	context1, _ := NewUcpContext(ucpParams)
	context2, _ := NewUcpContext(ucpParams)
	defer context1.Close()
	defer context2.Close()

	worker1, _ := context1.NewWorker(ucpWorkerParams)
	worker2, _ := context2.NewWorker(ucpWorkerParams.SetClientId(clientId))
	defer worker1.Close()
	defer worker2.Close()

	listener, _ := worker1.NewListener(listenerParams)
	defer listener.Close()

	listenerAttrs, _ := listener.Query(UCP_LISTENER_ATTR_FIELD_SOCKADDR)

	listenerAddress := listenerAttrs.Address

	t.Logf("Started listener on address: %v", listenerAddress)

	epParams.SetSocketAddress(listenerAddress)

	ep, err := worker2.NewEndpoint(epParams)
	if err != nil {
		t.Fatalf("Can't create endpoint %v", err)
	}

	flushReq1, _ := ep.FlushNonBlocking(nil)
	defer flushReq1.Close()
	for connReq == nil {
		worker1.Progress()
		worker2.Progress()
	}

	replyEpParams := (&UcpEpParams{}).SetConnRequest(connReq)
	replyEpParams.SetPeerErrorHandling().SetErrorHandler(logErrorHandler)
	replyEp, _ := worker1.NewEndpoint(replyEpParams)
	flushReq2, _ := replyEp.FlushNonBlocking(nil)
	defer flushReq2.Close()

	for flushReq2.GetStatus() == UCS_INPROGRESS {
		worker1.Progress()
		worker2.Progress()
	}

	epParams.SetErrorHandler(func(ep *UcpEp, status UcsStatus) {
		if status != UCS_ERR_REJECTED {
			t.Fatalf("Status is not rejected %v", status)
		}
		rejected = true
	})

	rejectEp, err := worker2.NewEndpoint(epParams)
	if err != nil {
		t.Fatalf("Can't create endpoint %v", err)
	}

	flushReq3, _ := rejectEp.FlushNonBlocking(nil)
	defer flushReq3.Close()

	for !rejected {
		worker1.Progress()
		worker2.Progress()
	}

	sendData := "listener test"
	sendMem := CBytes([]byte(sendData))
	defer FreeNativeMemory(sendMem)
	receiveMem := AllocateNativeMemory(4096)
	defer FreeNativeMemory(receiveMem)

	recvReq, _ := worker2.RecvTagNonBlocking(receiveMem, 4096, 1, 1, nil)
	defer recvReq.Close()
	replyEp.SendTagNonBlocking(1, sendMem, uint64(len(sendData)), nil)

	for recvReq.GetStatus() != UCS_OK {
		worker1.Progress()
		worker2.Progress()
	}

	closeReq, _ := ep.CloseNonBlockingForce(nil)
	closeReply, _ := replyEp.CloseNonBlockingForce(nil)

	for (closeReq.GetStatus() != UCS_OK) || (closeReply.GetStatus() != UCS_OK) {
		worker1.Progress()
		worker2.Progress()
	}
}
