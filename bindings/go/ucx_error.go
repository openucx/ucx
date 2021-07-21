/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucp

// #include <ucs/type/status.h>
import "C"

type UcxError struct {
	msg    string
	status UcsStatus
}

func NewUcxError(status C.ucs_status_t) error {
	return &UcxError{
		msg:    C.GoString(C.ucs_status_string(status)),
		status: UcsStatus(status),
	}
}

func (e *UcxError) Error() string { return e.msg }

func (e *UcxError) GetStatus() UcsStatus { return e.status }
