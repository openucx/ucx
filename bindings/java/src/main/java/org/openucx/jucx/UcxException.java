/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

/**
 * Exception to be thrown from JNI and all UCX routines.
 */
public class UcxException extends RuntimeException {

    private int status;

    public UcxException() {
        super();
    }

    public UcxException(String message) {
        super(message);
    }

    public UcxException(String message, int status) {
        super(message);
        this.status = status;
    }

    /**
     * Status of exception to compare with {@link org.openucx.jucx.ucs.UcsConstants.STATUS}
     */
    public int getStatus() {
        return status;
    }
}
