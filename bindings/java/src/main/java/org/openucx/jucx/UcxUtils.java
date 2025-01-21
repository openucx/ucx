/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import java.nio.ByteBuffer;

public class UcxUtils {

    private UcxUtils() { }

    /**
     * Returns view of underlying memory region as a ByteBuffer.
     * @param address - address of start of memory region
     */
    public static ByteBuffer getByteBufferView(long address, long length) {
        return getByteBufferViewNative(address, length);
    }

    /**
     * Returns native address of the current position of a direct byte buffer.
     */
    public static long getAddress(ByteBuffer buffer) {
        return getAddressNative(buffer) + buffer.position();
    }

    private static native long getAddressNative(ByteBuffer buffer);
    private static native ByteBuffer getByteBufferViewNative(long address, long length);
}
