// +build !cuda

/*
 * Copyright (C) 2023, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package cuda

import (
	"errors"
)

func CudaSetDevice() error {
	return errors.New("cuda support is disabled")
}
