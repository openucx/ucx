// +build cuda

/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
