/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
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
