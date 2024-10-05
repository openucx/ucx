/*
 * Copyright (C) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package goucxtests

import (
	"testing"
	. "ucx"
)

func TestUcpRma(t *testing.T) {
	const sendData string = "Hello GO"

	for _, memType := range get_mem_types() {
		sender := prepareContext(t, (&UcpParams{}).EnableRMA().EnableTag())
		receiver := prepareContext(t, (&UcpParams{}).EnableRMA().EnableTag())
		t.Logf("Testing RMA %v -> %v", memType.senderMemType, memType.recvMemType)

		ucpWorkerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_MULTI)

		receiver.worker, _ = receiver.context.NewWorker(ucpWorkerParams)
		sender.worker, _ = sender.context.NewWorker(ucpWorkerParams)
		connect(sender, receiver)

		sendMem := memoryAllocate(sender, uint64(len(sendData)), memType.senderMemType)
		memorySet(sender, []byte(sendData))

		receiveMem := memoryAllocate(receiver, 4096, memType.recvMemType)

		rkeyBuf, _ := receiver.mem.Pack()
		rkey, _ := sender.ep.Unpack(rkeyBuf)
		rkeyBuf.Close()

		sendRequest, _ := sender.ep.RmaPut(sendMem, uint64(len(sendData)), uint64(uintptr(receiveMem)), rkey, &UcpRequestParams{
			Cb: func(request *UcpRequest, status UcsStatus) {
				if status != UCS_OK {
					t.Fatalf("Request failed with status: %d", status)
				}

				request.Close()
			}})

		for sendRequest.GetStatus() == UCS_INPROGRESS {
			sender.worker.Progress()
			receiver.worker.Progress()
		}
		if recvString := string(memoryGet(receiver)[:len(sendData)]); recvString != sendData {
			t.Fatalf("Send data %s != recv data %s", sendData, recvString)
		}

		closeReq, _ := sender.ep.CloseNonBlockingFlush(nil)
		for closeReq.GetStatus() == UCS_INPROGRESS {
			sender.worker.Progress()
			receiver.worker.Progress()
		}
		closeReq.Close()
		rkey.Close()

		sender.Close()
		receiver.Close()
	}
}
