/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package goucxtests

import (
	"testing"
	. "ucx"
)

func TestUcpEpTag(t *testing.T) {
	const sendData string = "Hello GO"
	ucpParams := (&UcpParams{}).EnableTag().EnableWakeup()
	ucpWorkerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_SINGLE)
	ucpWorkerParams.WakeupTagSend().WakeupTagRecv()
	ucpContext1, _ := NewUcpContext(ucpParams)
	defer ucpContext1.Close()
	ucpContext2, _ := NewUcpContext(ucpParams)
	defer ucpContext2.Close()

	receiverWorker, _ := ucpContext1.NewWorker(ucpWorkerParams)
	defer receiverWorker.Close()
	workerAddress, _ := receiverWorker.GetAddress()
	defer workerAddress.Close()

	senderWorker, _ := ucpContext2.NewWorker(ucpWorkerParams)
	defer senderWorker.Close()

	epParams := (&UcpEpParams{}).SetUcpAddress(workerAddress)
	epParams.SetPeerErrorHandling().SetErrorHandler(func(ep *UcpEp, status UcsStatus) {
		t.Logf("Error handler called with status %d", status)
	})
	senderEp, _ := senderWorker.NewEndpoint(epParams)

	flushRequest, _ := senderEp.FlushNonBlocking(nil)

	for flushRequest.GetStatus() == UCS_INPROGRESS {
		senderWorker.Progress()
		receiverWorker.Progress()
	}

	flushRequest.Close()

	sendMem := CBytes(sendData)
	defer FreeNativeMemory(sendMem)
	receiveMem := AllocateNativeMemory(4096)
	defer FreeNativeMemory(receiveMem)

	recvRequest, _ := receiverWorker.RecvTagNonBlocking(receiveMem, 4096, 1, 1, &UcpRequestParams{
		Cb: func(request *UcpRequest, status UcsStatus, tagInfo *UcpTagRecvInfo) {

			if status != UCS_OK {
				t.Fatalf("Request failed with status: %d", status)
			}

			if tagInfo.Length != uint64(len(sendData)) {
				t.Fatalf("Data length %d != received length %d", len(sendData), tagInfo.Length)
			}

			request.Close()
		}})

	sendRequest, _ := senderEp.SendTagNonBlocking(1, sendMem, uint64(len(sendData)), &UcpRequestParams{
		Cb: func(request *UcpRequest, status UcsStatus) {
			if status != UCS_OK {
				t.Fatalf("Request failed with status: %d", status)
			}

			request.Close()
		}})

	for (sendRequest.GetStatus() == UCS_INPROGRESS) || (recvRequest.GetStatus() == UCS_INPROGRESS) {
		senderWorker.Progress()
		receiverWorker.Progress()
	}

	if recvString := string(GoBytes(receiveMem, uint64(len(sendData)))); recvString != sendData {
		t.Fatalf("Send data %s != recv data %s", sendData, recvString)
	}

	closeReq, _ := senderEp.CloseNonBlockingFlush(nil)
	defer closeReq.Close()

	for closeReq.GetStatus() == UCS_INPROGRESS {
		senderWorker.Progress()
		receiverWorker.Progress()
	}
}
