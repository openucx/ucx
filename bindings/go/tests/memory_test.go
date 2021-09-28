/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package goucxtests

import (
	"flag"
	"fmt"
	"testing"
	. "ucx"
	"unsafe"
)

var maxSize = flag.Uint64("s", 10_000_000, "Max size of memory to mmap. Default: 10M")

func memoryMap(context *UcpContext, address unsafe.Pointer, size uint64,
	memoryType UcsMemoryType) (*UcpMemory, error) {
	mmapParams := &UcpMmapParams{}
	mmapParams.SetAddress(address).SetLength(size).SetMemoryType(memoryType)

	return context.MemMap(mmapParams)
}

func memAlloc(context *UcpContext, size uint64, memoryType UcsMemoryType) (*UcpMemory, error) {
	mmapParams := &UcpMmapParams{}
	mmapParams.Allocate().SetLength(size).SetMemoryType(memoryType)

	return context.MemMap(mmapParams)
}

func BenchmarkUcpMmap(b *testing.B) {
	ucpParams := &UcpParams{}
	ucpParams.EnableTag()

	context, err := NewUcpContext(ucpParams)
	defer context.Close()
	memTypeMask, _ := context.MemoryTypesMask()

	if err != nil {
		b.Fatalf("Failed to create a context %v", err)
	}

	for i := 0; i < b.N; i++ {
		var size uint64 = 1024
		for size < *maxSize {
			b.Run(fmt.Sprintf("Allocate host memory %d", size), func(b *testing.B) {
				allocatedMemory, err := memAlloc(context, size, UCS_MEMORY_TYPE_HOST)

				if err != nil {
					b.Fatalf("Failed to allocate memory %v", err)
				}

				allocatedMemory.Close()
			})
			size = size << 1
		}
	}

	if IsMemTypeSupported(UCS_MEMORY_TYPE_CUDA, memTypeMask) {
		for i := 0; i < b.N; i++ {
			var size uint64 = 1024
			for size < *maxSize {
				b.Run(fmt.Sprintf("Allocate GPU memory %d", size), func(b *testing.B) {
					gpuMemory, err := memAlloc(context, size, UCS_MEMORY_TYPE_CUDA)

					if err != nil {
						b.Fatalf("Failed to allocate GPU memory %v", err)
					}

					gpuMemory.Close()
				})
				size = size << 1
			}
		}
	}
}

func TestUcpMmap(t *testing.T) {
	const testMemorySize uint64 = 1024
	ucpParams := &UcpParams{}
	ucpParams.EnableTag()

	context, err := NewUcpContext(ucpParams)
	defer context.Close()

	if err != nil {
		t.Fatalf("Failed to create a context %v", err)
	}

	allocatedMemory, err := memAlloc(context, testMemorySize, UCS_MEMORY_TYPE_HOST)

	if err != nil {
		t.Fatalf("Failed to allocate memory %v", err)
	}

	mmapAttrs, _ := allocatedMemory.Query(UCP_MEM_ATTR_FIELD_ADDRESS, UCP_MEM_ATTR_FIELD_LENGTH, UCP_MEM_ATTR_FIELD_MEM_TYPE)

	if UcsMemoryType(mmapAttrs.MemType) != UCS_MEMORY_TYPE_HOST {
		t.Fatalf("Allocated memory type is not host")
	}

	allocatedMemory.Close()
	nativeMemory := AllocateNativeMemory(testMemorySize)
	mapedMemory, err := memoryMap(context, nativeMemory, testMemorySize, UCS_MEMORY_TYPE_HOST)

	if err != nil {
		t.Fatalf("Failed to map memory %v", err)
	}

	mapedMemory.Close()
	FreeNativeMemory(nativeMemory)
	memTypeMask, _ := context.MemoryTypesMask()

	if IsMemTypeSupported(UCS_MEMORY_TYPE_CUDA, memTypeMask) {
		gpuMemory, err := memAlloc(context, testMemorySize, UCS_MEMORY_TYPE_CUDA)

		if err != nil {
			t.Fatalf("Failed to allocate GPU memory %v", gpuMemory)
		}

		gpuMemory.Close()
	}
}
