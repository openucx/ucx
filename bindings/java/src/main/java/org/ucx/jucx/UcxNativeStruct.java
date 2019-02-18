/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

/**
 * Wrapper around native ucx struct, that holds pointer address.
 */
public abstract class UcxNativeStruct {
    private Long nativeId;

    /**
     * Getter for native pointer as long.
     * @return long integer representing native pointer
     */
    public Long getNativeId() {
        return nativeId;
    }

    protected void setNativeId(Long nativeId) {
        this.nativeId = nativeId;
    }
}
