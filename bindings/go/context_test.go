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
	ucpParams.SetTagSenderMask(9).EnableStream().SetName("GO_Test").SetEstimatedNumPPN(1)

	context, err := NewUcpContext(ucpParams)

	if err != nil {
		t.Fatalf("Failed to create a context %v", err)
	}

	context.Close()

	if context.context != nil {
		t.Fatalf("Context not nil")
	}
}
