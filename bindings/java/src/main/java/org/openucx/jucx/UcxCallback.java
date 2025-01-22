/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import org.openucx.jucx.ucp.UcpRequest;

/**
 * Callback wrapper to notify successful or failure events from JNI.
 */

public class UcxCallback {
    public void onSuccess(UcpRequest request) {}

    public void onError(int ucsStatus, String errorMsg) {
        throw new UcxException(errorMsg);
    }
}
