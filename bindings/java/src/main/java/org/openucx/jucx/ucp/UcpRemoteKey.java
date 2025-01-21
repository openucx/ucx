/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxNativeStruct;

import java.io.Closeable;

/**
 * Remote memory handle is an opaque object representing remote memory access
 * information. Typically, the handle includes a memory access key and other
 * network hardware specific information, which are input to remote memory
 * access operations, such as PUT, GET, and ATOMIC. The object is
 * communicated to remote peers to enable an access to the memory region.
 */
public class UcpRemoteKey extends UcxNativeStruct implements Closeable {

    /**
     * Private constructor to construct from JNI only.
     */
    private UcpRemoteKey() {

    }

    private UcpRemoteKey(long nativeRkeyPtr) {
        setNativeId(nativeRkeyPtr);
    }

    @Override
    public void close() {
        rkeyDestroy(getNativeId());
        setNativeId(null);
    }

    private static native void rkeyDestroy(long ucpRkeyId);
}
