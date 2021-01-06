/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

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

    private void setNativeId(long nativeId) {
        if (nativeId > 0) {
            this.nativeId = nativeId;
        } else {
            this.nativeId = null;
        }
    }

    protected void setNativeId(Long nativeId) {
        if (nativeId != null && nativeId < 0) {
            throw new UcxException("UcxNativeStruct.setNativeId: invalid native pointer: "
                + nativeId);
        }
        this.nativeId = nativeId;
    }
}
