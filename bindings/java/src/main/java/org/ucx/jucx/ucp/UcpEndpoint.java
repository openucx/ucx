/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.ucp;

import org.ucx.jucx.UcxCallback;
import org.ucx.jucx.UcxException;
import org.ucx.jucx.UcxNativeStruct;

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

    public void getNonBlocking(long remoteAddress,  UcpRemoteKey remoteKey,
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
        getNonBlockingNative(getNativeId(), remoteAddress, remoteKey.getNativeId(),
            dst, callback);
    }

    private static native long createEndpointNative(UcpEndpointParams params, long workerId);

    private static native void destroyEndpointNative(long epId);

    private static native UcpRemoteKey unpackRemoteKey(long epId, ByteBuffer rkeyBuffer);

    private static native void getNonBlockingNative(long enpointId,
                                                    long remoteAddress,
                                                    long ucpRkeyId,
                                                    ByteBuffer localData,
                                                    UcxCallback callback);
}
