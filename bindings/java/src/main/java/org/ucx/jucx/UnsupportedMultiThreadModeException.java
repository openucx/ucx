/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

/**
 * Thrown to indicate that multithread mode was set but is not supported.
 */
public class UnsupportedMultiThreadModeException extends UnsupportedOperationException {
    /**
     * Constructs an {@link UnsupportedMultiThreadModeException} with no detail message.
     */
    public UnsupportedMultiThreadModeException() {}

    private static final long serialVersionUID = -6459116594056547143L;
}
