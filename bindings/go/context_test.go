/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package ucp

import (
	"testing"
)

func TestUcpContext(t *testing.T) {
	ucpParams := &UcpParams{}
	ucpParams.SetTagSenderMask(9).EnableStream()

	context, err := NewUcpContext(ucpParams)

	if err != nil {
		t.Fatalf("failed to create a context %v", err)
	}

	context.Close()

	if context.context !=nil {
		t.Fatalf("context not nil")
	}
}
