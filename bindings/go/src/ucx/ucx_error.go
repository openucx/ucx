/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

package ucx

// #include <ucs/type/status.h>
import "C"

type UcxError struct {
	msg    string
	status UcsStatus
}

func NewUcxError(status UcsStatus) error {
	return &UcxError{
		msg:    C.GoString(C.ucs_status_string(C.ucs_status_t(status))),
		status: status,
	}
}

func newUcxError(status C.ucs_status_t) error {
	return &UcxError{
		msg:    C.GoString(C.ucs_status_string(status)),
		status: UcsStatus(status),
	}
}

func (e *UcxError) Error() string { return e.msg }

func (e *UcxError) GetStatus() UcsStatus { return e.status }
