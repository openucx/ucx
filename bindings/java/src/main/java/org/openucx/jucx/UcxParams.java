/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

/**
 * Common interface for representing parameters to instantiate ucx objects.
 */
public abstract class UcxParams {
    /**
     * Mask of valid fields in this structure.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    protected long fieldMask;
    /**
     * Reset state of parameters.
     */
    public UcxParams clear() {
        fieldMask = 0L;
        return this;
    }

    public static void checkArraySizes(long[] array1, long[] array2) {
        if (array1.length != array2.length) {
            throw new UcxException("Arrays of not equal sizes: " +
                array1.length + " != " + array2.length);
        }
    }
}
