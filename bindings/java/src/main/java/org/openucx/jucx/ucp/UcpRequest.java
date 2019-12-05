/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxNativeStruct;

import java.io.Closeable;

/**
 * Request object, that returns by ucp operations (GET, PUT, SEND, etc.).
 * Call {@link UcpRequest#isCompleted()} to monitor completion of request.
 */
public class UcpRequest extends UcxNativeStruct implements Closeable {

    private UcpRequest(long nativeId) {
        setNativeId(nativeId);
    }

    /**
     * @return whether this request is completed.
     */
    public boolean isCompleted() {
        return (getNativeId() == null) || isCompletedNative(getNativeId());
    }

    /**
     * This routine releases the non-blocking request back to the library, regardless
     * of its current state. Communications operations associated with this request
     * will make progress internally, however no further notifications or callbacks
     * will be invoked for this request.
     */
    @Override
    public void close() {
        if (getNativeId() != null) {
            closeRequestNative(getNativeId());
        }
    }

    private static native boolean isCompletedNative(long ucpRequest);

    private static native void closeRequestNative(long ucpRequest);
}
