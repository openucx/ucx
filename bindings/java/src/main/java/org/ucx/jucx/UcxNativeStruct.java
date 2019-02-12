/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

/**
 * Wrapper around native ucx struct, that holds pointer address.
 */
public abstract class UcxNativeStruct {
    private long nativeId;

    public UcxNativeStruct(long nativeId) {
        this.nativeId = nativeId;
    }

    /**
     * Getter for native pointer as long.
     * @return long integer representing native pointer
     */
    public long getNativeId() {
        return nativeId;
    }

    protected void setNativeId(long nativeId) {
        this.nativeId = nativeId;
    }
}
