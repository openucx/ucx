/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucp

// #include <stdlib.h>
import "C"
import "unsafe"

func AllocateNativeMemory(size uint64) unsafe.Pointer {
	return C.malloc(C.ulong(size))
}

func FreeNativeMemory(pointer unsafe.Pointer) {
	C.free(pointer)
}
