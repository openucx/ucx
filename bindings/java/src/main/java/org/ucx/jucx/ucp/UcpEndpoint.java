/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.ucp;

import org.ucx.jucx.UcxCallback;
import org.ucx.jucx.UcxException;
import org.ucx.jucx.UcxNativeStruct;
import org.ucx.jucx.UcxRequest;

import java.io.Closeable;
import java.nio.ByteBuffer;

public class UcpEndpoint extends UcxNativeStruct implements Closeable {

    public UcpEndpoint(UcpWorker worker, UcpEndpointParams params) {
        setNativeId(createEndpointNative(params, worker.getNativeId()));
    }

    @Override
    public void close() {
        destroyEndpointNative(getNativeId());
        setNativeId(null);
    }

    /**
     * This routine unpacks the remote key (RKEY) object into the local memory
     * such that it can be accessed and used by UCP routines.
     * @param rkeyBuffer - Packed remote key buffer
     *                     (see {@link UcpMemory#getRemoteKeyBuffer()}).
     */
    public UcpRemoteKey unpackRemoteKey(ByteBuffer rkeyBuffer) {
        return unpackRemoteKey(getNativeId(), rkeyBuffer);
    }

    /**
     * Non-blocking remote memory get operation.
     * This routine initiates a load of a contiguous block of data that is
     * described by the remote memory address {@code remoteAddress} and the
     * {@code remoteKey} "memory handle". The routine returns immediately and <strong>does</strong>
     * not guarantee that remote data is loaded and stored under the local {@code dst} buffer.
     * {@code callback} is invoked on completion of this operation.
     * @return {@link UcxRequest} object that can be monitored for completion.
     */
    public UcxRequest getNonBlocking(long remoteAddress, UcpRemoteKey remoteKey,
                                     ByteBuffer dst, UcxCallback callback) {
        if (!dst.isDirect()) {
            throw new UcxException("Data buffer must be direct.");
        }
        if (remoteKey.getNativeId() == null) {
            throw new UcxException("Remote key is null.");
        }
        if (callback == null) {
            callback = new UcxCallback();
        }
        return getNonBlockingNative(getNativeId(), remoteAddress, remoteKey.getNativeId(),
            dst, callback);
    }

    /**
     * Non-blocking tagged-send operations
     * This routine sends a messages that is described by the local buffer {@code sendBuffer},
     * to the destination endpoint. Each message is associated with a {@code tag} value
     * that is used for message matching on the
     * {@link UcpWorker#recvNonBlocking(ByteBuffer, long, UcxCallback)}
     * "receiver".  The routine is non-blocking and therefore returns immediately,
     * however the actual send operation may be delayed.
     * The send operation is considered completed when  it is safe to reuse the source
     * {@code data} buffer. {@code callback} is invoked on completion of this operation.
     */
    public UcxRequest sendNonBlocking(ByteBuffer sendBuffer, long tag, UcxCallback callback) {
        if (!sendBuffer.isDirect()) {
            throw new UcxException("Send buffer must be direct.");
        }
        if (callback == null) {
            callback = new UcxCallback();
        }
        return sendNonBlockingNative(getNativeId(), sendBuffer, tag, callback);
    }

    /**
     * Non blocking send operation. Invokes
     * {@link UcpEndpoint#sendNonBlocking(ByteBuffer, long, UcxCallback)} with default 0 tag.
     */
    public UcxRequest sendNonBlocking(ByteBuffer sendBuffer, UcxCallback callback) {
        return sendNonBlocking(sendBuffer, 0, callback);
    }

    /**
     * Non-blocking remote memory put operation.
     * This routine initiates a storage of contiguous block of data that is
     * described by the local {@code data} buffer in the remote contiguous memory
     * region described by {@code remoteAddress} address and the {@code remoteKey} "memory
     * handle". The routine returns immediately and <strong>does</strong> not
     * guarantee re-usability of the source {@code data} buffer.
     * {@code callback} is invoked on completion of this operation.
     */
    public UcxRequest putNonBlocking(ByteBuffer src, long remoteAddress, UcpRemoteKey remoteKey,
                                     UcxCallback callback) {
        if (!src.isDirect()) {
            throw new UcxException("Data buffer must be direct.");
        }
        if (remoteKey.getNativeId() == null) {
            throw new UcxException("Remote key is null.");
        }
        if (callback == null) {
            callback = new UcxCallback();
        }
        return putNonBlockingNative(getNativeId(), src, remoteAddress,
            remoteKey.getNativeId(), callback);
    }

    private static native long createEndpointNative(UcpEndpointParams params, long workerId);

    private static native void destroyEndpointNative(long epId);

    private static native UcpRemoteKey unpackRemoteKey(long epId, ByteBuffer rkeyBuffer);

    private static native UcxRequest getNonBlockingNative(long enpointId,
                                                          long remoteAddress,
                                                          long ucpRkeyId,
                                                          ByteBuffer localData,
                                                          UcxCallback callback);

    private static native UcxRequest sendNonBlockingNative(long enpointId, ByteBuffer sendBuffer,
                                                           long tag,
                                                           UcxCallback callback);

    private static native UcxRequest putNonBlockingNative(long enpointId, ByteBuffer src,
                                                          long remoteAddr, long ucpRkeyId,
                                                          UcxCallback callback);
}
