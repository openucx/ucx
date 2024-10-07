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
	const data string = "Hello GO"
	const length uint64 = uint64(len(data))

	for _, memType := range get_mem_types() {
		requester := prepareContext(t, (&UcpParams{}).EnableRMA().EnableTag())
		responder := prepareContext(t, (&UcpParams{}).EnableRMA().EnableTag())
		t.Logf("Testing RMA %v -> %v", memType.senderMemType, memType.recvMemType)

		ucpWorkerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_MULTI)

		requester.worker, _ = requester.context.NewWorker(ucpWorkerParams)
		responder.worker, _ = responder.context.NewWorker(ucpWorkerParams)
		connect(requester, responder)

		localMem := memoryAllocate(requester, length, memType.senderMemType)
		memorySet(requester, []byte(data))

		remoteMem := memoryAllocate(responder, 4096, memType.recvMemType)

		rkeyBuf, _ := responder.mem.Pack()
		rkey, _ := requester.ep.Unpack(rkeyBuf)
		rkeyBuf.Close()

		putRequest, _ := requester.ep.RmaPut(localMem, length, uint64(uintptr(remoteMem)), rkey, &UcpRequestParams{
			Cb: func(request *UcpRequest, status UcsStatus) {
				if status != UCS_OK {
					t.Fatalf("Request failed with status: %d", status)
				}

				request.Close()
			}})

		for putRequest.GetStatus() == UCS_INPROGRESS {
			requester.worker.Progress()
			responder.worker.Progress()
		}

		if remoteData := string(memoryGet(responder)[:length]); remoteData != data {
			t.Fatalf("Remote data %s != data %s", remoteData, data)
		}

		memorySet(requester, make([]byte, length))

		getRequest, _ := requester.ep.RmaGet(localMem, length, uint64(uintptr(remoteMem)), rkey, &UcpRequestParams{
			Cb: func(request *UcpRequest, status UcsStatus) {
				if status != UCS_OK {
					t.Fatalf("Request failed with status: %d", status)
				}

				request.Close()
			}})

		for getRequest.GetStatus() == UCS_INPROGRESS {
			requester.worker.Progress()
			responder.worker.Progress()
		}

		if localData := string(memoryGet(responder)[:length]); localData != data {
			t.Fatalf("Local data %s != data %s", localData, data)
		}

		closeReq, _ := requester.ep.CloseNonBlockingFlush(nil)
		for closeReq.GetStatus() == UCS_INPROGRESS {
			requester.worker.Progress()
			responder.worker.Progress()
		}
		closeReq.Close()
		rkey.Close()

		requester.Close()
		responder.Close()
	}
}
