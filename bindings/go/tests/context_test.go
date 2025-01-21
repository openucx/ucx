/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

package goucxtests

import (
	"testing"
	. "github.com/openucx/ucx/bindings/go/src/ucx"
)

func TestUcpContext(t *testing.T) {
	ucpParams := &UcpParams{}
	ucpParams.SetTagSenderMask(9).EnableStream().SetName("GO_Test").SetEstimatedNumPPN(1)

	context, err := NewUcpContext(ucpParams)

	if err != nil {
		t.Fatalf("Failed to create a context %v", err)
	}

	ucpParams.SetName("Go test2")

	context.Close()
}
