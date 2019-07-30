/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

/**
 * Exception to be thrown from JNI and all UCX routines.
 */
public class UcxException extends RuntimeException {

    public UcxException() {
        super();
    }

    public UcxException(String message) {
        super(message);
    }
}
