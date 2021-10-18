/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package goucxtests

import (
	. "ucx"
	"unsafe"
)

const selfEpTag uint64 = ^uint64(0)

func memoryAllocate(entity *TestEntity, size uint64, memoryType UcsMemoryType) unsafe.Pointer {
	mmapParams := &UcpMmapParams{}
	mmapParams.Allocate().SetLength(size).SetMemoryType(memoryType)

	result, err := entity.context.MemMap(mmapParams)
	if err != nil {
		entity.t.Fatalf("Failed to allocate memory %v", err)
	}

	entity.mem = result

	memAttr, _ := entity.mem.Query(UCP_MEM_ATTR_FIELD_ADDRESS)

	return memAttr.Address
}

func createSelfEp(entity *TestEntity) {
	if entity.selfEp == nil {
		ucpAddress, err := entity.worker.GetAddress()
		if err != nil {
			entity.t.Fatalf("Failed to get address %v", err)
		}
		entity.selfEp, err = entity.worker.NewEndpoint((&UcpEpParams{}).SetUcpAddress(ucpAddress))
		if err != nil {
			entity.t.Fatalf("Failed to create endpoint %v", err)
		}
	}
}

func sendRecv(entity *TestEntity, sendAddress unsafe.Pointer, sendLength uint64, sendMemType UcsMemoryType,
	recvAddr unsafe.Pointer, recvLength uint64, recvMemoryType UcsMemoryType) {

	createSelfEp(entity)
	recvRequest, _ := entity.worker.RecvTagNonBlocking(recvAddr, recvLength, selfEpTag, selfEpTag, (&UcpRequestParams{}).SetMemType(recvMemoryType))
	defer recvRequest.Close()

	sendReq, _ := entity.selfEp.SendTagNonBlocking(selfEpTag, sendAddress, sendLength, (&UcpRequestParams{}).SetMemType(sendMemType))

	defer sendReq.Close()

	for recvRequest.GetStatus() == UCS_INPROGRESS {
		entity.worker.Progress()
	}
}

func memorySet(entity *TestEntity, data []byte) {
	memAttr, _ := entity.mem.Query(UCP_MEM_ATTR_FIELD_ADDRESS, UCP_MEM_ATTR_FIELD_LENGTH, UCP_MEM_ATTR_FIELD_MEM_TYPE)

	stringBytes := CBytes(data)
	defer FreeNativeMemory((stringBytes))
	sendRecv(entity, stringBytes, uint64(len(data)), UCS_MEMORY_TYPE_HOST, memAttr.Address, memAttr.Length, memAttr.MemType)
}

func memoryGet(entity *TestEntity) []byte {
	memAttr, _ := entity.mem.Query(UCP_MEM_ATTR_FIELD_ADDRESS, UCP_MEM_ATTR_FIELD_LENGTH, UCP_MEM_ATTR_FIELD_MEM_TYPE)

	if memAttr.MemType == UCS_MEMORY_TYPE_HOST {
		return GoBytes(memAttr.Address, memAttr.Length)
	} else {
		recvMem := AllocateNativeMemory(memAttr.Length)
		sendRecv(entity, memAttr.Address, memAttr.Length, memAttr.MemType, recvMem, memAttr.Length, UCS_MEMORY_TYPE_HOST)
		return GoBytes(recvMem, memAttr.Length)
	}
}

// Progress thread that progress a worker until it receives quit signal from channel.
func progressThread(quit chan bool, worker *UcpWorker) {
	for {
		select {
		case <-quit:
			close(quit)
			return
		default:
			worker.Progress()
		}
	}
}
