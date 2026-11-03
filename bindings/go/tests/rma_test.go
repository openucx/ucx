/*
 * Copyright (C) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package goucxtests

import (
	"testing"
	"unsafe"
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

		testOp := func(op func(buffer unsafe.Pointer, size uint64, remote_addr uint64, rkey *UcpRKey, params *UcpRequestParams) (*UcpRequest, error)) {
			req, _ := op(localMem, length, uint64(uintptr(remoteMem)), rkey, &UcpRequestParams{
			Cb: func(request *UcpRequest, status UcsStatus) {
				if status != UCS_OK {
					t.Fatalf("Request failed with status: %d", status)
				}

				request.Close()
			}})

			for req.GetStatus() == UCS_INPROGRESS {
				requester.worker.Progress()
				responder.worker.Progress()
			}

			if xferData := string(memoryGet(responder)[:length]); xferData != data {
				t.Fatalf("Transferred data %s != data %s", xferData, data)
			}
		}

		testOp(requester.ep.RmaPut)

		memorySet(requester, make([]byte, length))

		testOp(requester.ep.RmaGet)

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
