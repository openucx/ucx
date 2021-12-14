/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package goucxtests

import (
	"errors"
	"fmt"
	"testing"
	. "ucx"
	"unsafe"
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

func prepareContext(t *testing.T, ucpParams *UcpParams) *TestEntity {
	if ucpParams == nil {
		ucpParams = (&UcpParams{}).EnableAM().EnableTag().EnableWakeup()
	}

	context, err := NewUcpContext(ucpParams)

	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}

	return &TestEntity{
		t:       t,
		context: context,
	}
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

func connect(sender *TestEntity, receiver *TestEntity) {
	workerAddress, _ := receiver.worker.GetAddress()
	epParams := (&UcpEpParams{}).SetUcpAddress(workerAddress)
	epParams.SetPeerErrorHandling().SetErrorHandler(func(ep *UcpEp, status UcsStatus) {
		sender.t.Logf("Error handler called with status %d", status)
	})

	sender.ep, _ = sender.worker.NewEndpoint(epParams)
	workerAddress.Close()
	flushRequest, _ := sender.ep.FlushNonBlocking(nil)

	for flushRequest.GetStatus() == UCS_INPROGRESS {
		sender.worker.Progress()
		receiver.worker.Progress()
	}

	flushRequest.Close()
}

func TestUcpEpTag(t *testing.T) {
	const sendData string = "Hello GO"

	for _, memType := range get_mem_types() {
		sender := prepareContext(t, nil)
		receiver := prepareContext(t, nil)
		t.Logf("Testing tag %v -> %v", memType.senderMemType, memType.recvMemType)

		ucpWorkerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_SINGLE)
		ucpWorkerParams.WakeupTagSend().WakeupTagRecv()

		receiver.worker, _ = receiver.context.NewWorker(ucpWorkerParams)
		sender.worker, _ = sender.context.NewWorker(ucpWorkerParams)
		connect(sender, receiver)

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

func TestUcpEpAm(t *testing.T) {
	const sendData string = "Hello GO AM"
	const dataLen uint64 = uint64(len(sendData))

	for _, memType := range get_mem_types() {
		sender := prepareContext(t, nil)
		receiver := prepareContext(t, nil)
		t.Logf("Testing AM %v -> %v", memType.senderMemType, memType.recvMemType)

		ucpWorkerParams := (&UcpWorkerParams{}).SetThreadMode(UCS_THREAD_MODE_MULTI)
		receiver.worker, _ = receiver.context.NewWorker(ucpWorkerParams)
		sender.worker, _ = sender.context.NewWorker(&UcpWorkerParams{})

		connect(sender, receiver)

		sendMem := memoryAllocate(sender, dataLen, memType.senderMemType)
		receiveMem := memoryAllocate(receiver, 4096, memType.recvMemType)
		memorySet(sender, []byte(sendData))

		// To pass from progress thread AM data and test it's content
		amDataChan := make(chan *UcpAmData, 1)
		// To keep all requests from threads and close them on completion
		requests := make(chan *UcpRequest, 4)
		// t.Fatalf can't be called from non main thread need to pass an error
		threadErr := make(chan error)

		// Test eager handler with data persistance
		receiver.worker.SetAmRecvHandler(1, UCP_AM_FLAG_WHOLE_MSG|UCP_AM_FLAG_PERSISTENT_DATA, func(header unsafe.Pointer, headerSize uint64,
			data *UcpAmData, replyEp *UcpEp) UcsStatus {
			if !data.IsDataValid() {
				threadErr <- errors.New("Data is not received")
			}

			if !data.CanPersist() {
				threadErr <- errors.New("Data descriptor can't be persisted")
			}

			headerData := string(GoBytes(header, headerSize))
			if headerData != sendData {
				threadErr <- fmt.Errorf("Header data %v != %v", headerData, sendData)
			}
			amDataChan <- data
			return UCS_INPROGRESS
		})

		// Test rndv handler
		receiver.worker.SetAmRecvHandler(2, UCP_AM_FLAG_WHOLE_MSG, func(header unsafe.Pointer, headerSize uint64,
			data *UcpAmData, replyEp *UcpEp) UcsStatus {
			if replyEp == nil {
				threadErr <- errors.New("Reply endpoint is not set")
			}

			recvRequest, _ := data.Receive(receiveMem, data.Length(), (&UcpRequestParams{}).SetMemType(memType.recvMemType))

			// Test reply ep functionality
			sendReq, _ := replyEp.SendAmNonBlocking(3, header, headerSize, nil, 0, UCP_AM_SEND_FLAG_EAGER, nil)
			requests <- recvRequest
			requests <- sendReq
			return UCS_OK
		})

		// To notify progress thread to exit
		quit := make(chan bool)
		go progressThread(quit, receiver.worker)

		headerMem := CBytes([]byte(sendData))
		sendChan := make(chan bool, 1)
		sender.worker.SetAmRecvHandler(3, UCP_AM_FLAG_WHOLE_MSG, func(header unsafe.Pointer, headerSize uint64,
			data *UcpAmData, replyEp *UcpEp) UcsStatus {
			str := string(GoBytes(header, headerSize))
			if str != sendData {
				t.Fatalf("Received data %v != %v", str, sendData)
			}
			sendChan <- true
			return UCS_OK
		})

		send1, _ := sender.ep.SendAmNonBlocking(1, headerMem, dataLen, sendMem, dataLen, UCP_AM_SEND_FLAG_EAGER, nil)
		send2, _ := sender.ep.SendAmNonBlocking(2, headerMem, dataLen, sendMem, dataLen, UCP_AM_SEND_FLAG_RNDV|UCP_AM_SEND_FLAG_REPLY, nil)

		requests <- send1
		requests <- send2

	senderProgress:
		for {
			select {
			case err := <-threadErr:
				t.Fatalf(err.Error())
			case <-sendChan:
				break senderProgress
			default:
				sender.worker.Progress()
			}
		}

		for len(requests) != cap(requests) {
		}
		close(requests)
		for req := range requests {
			for req.GetStatus() == UCS_INPROGRESS {
				sender.worker.Progress()
			}
			req.Close()
		}

		amData := <-amDataChan
		close(amDataChan)
		amDataAddr, _ := amData.DataPointer()
		data := string(GoBytes(amDataAddr, amData.Length()))
		if data != sendData {
			t.Fatalf("Received amData %v != %v", data, sendData)
		}

		data = string(memoryGet(receiver)[:len(sendData)])
		if data != sendData {
			t.Fatalf("Received data %v != %v", data, sendData)
		}

		amData.Close()
		quit <- true

		sender.Close()
		receiver.Close()
	}
}
