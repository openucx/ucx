/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxCallback;
import org.openucx.jucx.UcxNativeStruct;
import org.openucx.jucx.ucs.UcsConstants;

import java.io.Closeable;
import java.nio.ByteBuffer;

/**
 * Request object, that returns by ucp operations (GET, PUT, SEND, etc.).
 * Call {@link UcpRequest#isCompleted()} to monitor completion of request.
 */
public class UcpRequest extends UcxNativeStruct implements Closeable {

    private long recvSize;

    private long senderTag;

    private int status = UcsConstants.STATUS.UCS_INPROGRESS;

    private UcpRequest(long nativeId) {
        setNativeId(nativeId);
    }

    /**
     * To initialize for failed and immediately completed requests.
     */
    private UcpRequest() { }

    /**
     * The size of the received data in bytes, valid only for recv requests, e.g.:
     * {@link UcpWorker#recvTaggedNonBlocking(ByteBuffer buffer, UcxCallback clb)}
     */
    public long getRecvSize() {
        return recvSize;
    }

    /**
     * Sender tag, valid only for tag receive requests.
     */
    public long getSenderTag() {
        return senderTag;
    }

    /**
     * @return whether this request is completed.
     */
    public boolean isCompleted() {
        return (getNativeId() == null) || isCompletedNative(getNativeId());
    }

    /**
     * @return status of the current request
     */
    public int getStatus() {
        return status;
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
