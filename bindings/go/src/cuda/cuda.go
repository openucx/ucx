// +build cuda

/*
 * Copyright (C) 2023, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package cuda

// #include <cuda.h>
// #include <cuda_runtime.h>
import "C"
import (
	"errors"
)

func CudaSetDevice() error {
	var count C.int
	var ctx C.CUcontext

	if C.cuCtxGetCurrent(&ctx) == C.CUDA_SUCCESS && ctx != nil {
		return nil
	}

	if ret := C.cudaGetDeviceCount(&count); ret != C.cudaSuccess {
		return errors.New("Failed to get cuda device count")
	}

	if count < 1 {
		return errors.New("No compute-capable cuda devices")
	}

	if ret := C.cudaSetDevice(0); ret != C.cudaSuccess {
		return errors.New("Failed to set cuda device")
	}

	// cudaSetDevice initializes the primary context for the specified device
	// starting from CUDA12. It is required to call cudaDeviceSynchronize for
	// the older versions.
	if ret := C.cudaDeviceSynchronize(); ret != C.cudaSuccess {
		return errors.New("Failed to synchronize cuda device")
	}

	return nil
}
