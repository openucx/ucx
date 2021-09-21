/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package goucxtests

import (
	"testing"
	. "ucx"
)

type TestEntity struct {
	context *UcpContext
	worker  *UcpWorker
	ep      *UcpEp
	selfEp  *UcpEp
	mem     *UcpMemory
	t       *testing.T
}

type memTypePair struct {
	senderMemType UcsMemoryType
	recvMemType   UcsMemoryType
}

func (e *TestEntity) Close() {
	if e.mem != nil {
		e.mem.Close()
	}

	if e.selfEp != nil {
		closeReq, _ := e.selfEp.CloseNonBlockingForce(nil)
		for closeReq.GetStatus() == UCS_INPROGRESS {
			e.worker.Progress()
		}
		closeReq.Close()
	}

	if e.worker != nil {
		e.worker.Close()
	}

	if e.context != nil {
		e.context.Close()
	}

}

func prepare(t *testing.T) (*TestEntity, *TestEntity) {
	sender := &TestEntity{
		t: t,
	}
	receiver := &TestEntity{
		t: t,
	}
	ucpParams := (&UcpParams{}).EnableAM().EnableTag().EnableWakeup()

	sender.context, _ = NewUcpContext(ucpParams)
	receiver.context, _ = NewUcpContext(ucpParams)

	return sender, receiver
}

func get_mem_types() []memTypePair {
	ucpParams := (&UcpParams{}).EnableAM().EnableTag().EnableWakeup()
	context, _ := NewUcpContext(ucpParams)
	defer context.Close()
	memTypeMask, _ := context.MemoryTypesMask()
	memTypePairs := []memTypePair{memTypePair{UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_HOST}}

	if IsMemTypeSupported(UCS_MEMORY_TYPE_CUDA, memTypeMask) {
		memTypePairs = append(memTypePairs, memTypePair{UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_CUDA})
		memTypePairs = append(memTypePairs, memTypePair{UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_HOST})
		memTypePairs = append(memTypePairs, memTypePair{UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_CUDA})
	}
	return memTypePairs
}

func TestUcpEpTag(t *testing.T) {
	const sendData string = "Hello GO"

	for _, memType := range get_mem_types() {
		sender, receiver := prepare(t)
		t.Logf("Testing tag %v -> %v", memType.senderMemType, memType.recvMemType)

		ucpWorkerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_SINGLE)
		ucpWorkerParams.WakeupTagSend().WakeupTagRecv()

		receiver.worker, _ = receiver.context.NewWorker(ucpWorkerParams)
		workerAddress, _ := receiver.worker.GetAddress()

		sender.worker, _ = sender.context.NewWorker(ucpWorkerParams)
		epParams := (&UcpEpParams{}).SetUcpAddress(workerAddress)
		epParams.SetPeerErrorHandling().SetErrorHandler(func(ep *UcpEp, status UcsStatus) {
			t.Logf("Error handler called with status %d", status)
		})

		sender.ep, _ = sender.worker.NewEndpoint(epParams)
		workerAddress.Close()

		flushRequest, _ := sender.ep.FlushNonBlocking(nil)

		for flushRequest.GetStatus() == UCS_INPROGRESS {
			sender.worker.Progress()
			receiver.worker.Progress()
		}

		flushRequest.Close()

		sendMem := memoryAllocate(sender, uint64(len(sendData)), memType.senderMemType)
		memorySet(sender, []byte(sendData))

		receiveMem := memoryAllocate(receiver, 4096, memType.recvMemType)

		recvRequest, _ := receiver.worker.RecvTagNonBlocking(receiveMem, 4096, 1, 1, &UcpRequestParams{
			Cb: func(request *UcpRequest, status UcsStatus, tagInfo *UcpTagRecvInfo) {

				if status != UCS_OK {
					t.Fatalf("Request failed with status: %d", status)
				}

				if tagInfo.Length != uint64(len(sendData)) {
					t.Fatalf("Data length %d != received length %d", len(sendData), tagInfo.Length)
				}

				request.Close()
			}})

		sendRequest, _ := sender.ep.SendTagNonBlocking(1, sendMem, uint64(len(sendData)), &UcpRequestParams{
			Cb: func(request *UcpRequest, status UcsStatus) {
				if status != UCS_OK {
					t.Fatalf("Request failed with status: %d", status)
				}

				request.Close()
			}})

		for (sendRequest.GetStatus() == UCS_INPROGRESS) || (recvRequest.GetStatus() == UCS_INPROGRESS) {
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

		sender.Close()
		receiver.Close()
	}

}
