/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.*;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.stream.Stream;

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
        return unpackRemoteKey(getNativeId(),
                    UcxUtils.getAddress(rkeyBuffer));
    }

    private void checkRemoteAccessParams(ByteBuffer buf, UcpRemoteKey remoteKey) {
        if (!buf.isDirect()) {
            throw new UcxException("Data buffer must be direct.");
        }
        if (remoteKey.getNativeId() == null) {
            throw new UcxException("Remote key is null.");
        }
    }

    /**
     * Non-blocking remote memory put operation.
     * This routine initiates a storage of contiguous block of data that is
     * described by the local {@code src} buffer, starting of it's {@code src.position()}
     * and size {@code src.remaining()} in the remote contiguous memory
     * region described by {@code remoteAddress} address and the {@code remoteKey} "memory
     * handle". The routine returns immediately and <strong>does</strong> not
     * guarantee re-usability of the source {@code data} buffer.
     * {@code callback} is invoked on completion of this operation.
     */
    public UcxRequest putNonBlocking(ByteBuffer src, long remoteAddress, UcpRemoteKey remoteKey,
                                     UcxCallback callback) {

        checkRemoteAccessParams(src, remoteKey);

        return putNonBlockingNative(getNativeId(), UcxUtils.getAddress(src),
            src.remaining(), remoteAddress,
            remoteKey.getNativeId(), callback);
    }

    public UcxRequest putNonBlocking(long localAddress, long size,
                                     long remoteAddress, UcpRemoteKey remoteKey,
                                     UcxCallback callback) {

        return putNonBlockingNative(getNativeId(), localAddress,
            size, remoteAddress, remoteKey.getNativeId(), callback);
    }

    /**
     * Non-blocking remote memory get operation.
     * This routine initiates a load of a contiguous block of data that is
     * described by the remote memory address {@code remoteAddress} and the
     * {@code remoteKey} "memory handle". The routine returns immediately and <strong>does</strong>
     * not guarantee that remote data is loaded and stored under the local {@code dst} buffer
     * starting of it's {@code dst.position()} and size {@code dst.remaining()}.
     * {@code callback} is invoked on completion of this operation.
     * @return {@link UcxRequest} object that can be monitored for completion.
     */
    public UcxRequest getNonBlocking(long remoteAddress, UcpRemoteKey remoteKey,
                                     ByteBuffer dst, UcxCallback callback) {

        checkRemoteAccessParams(dst, remoteKey);

        return getNonBlockingNative(getNativeId(), remoteAddress, remoteKey.getNativeId(),
            UcxUtils.getAddress(dst), dst.remaining(), callback);
    }

    public UcxRequest getNonBlocking(long remoteAddress, UcpRemoteKey remoteKey,
                                     long localAddress, long size, UcxCallback callback) {

        return getNonBlockingNative(getNativeId(), remoteAddress, remoteKey.getNativeId(),
            localAddress, size, callback);
    }

    // Non-blocking implicit remote memory get operations.

    /**
     * Non-blocking implicit remote memory get operation.
     * This routine initiate a load of contiguous block of data that is described
     * by the remote memory address {@code remoteAddress} and the
     * {@code remoteKey} "memory handle" in the local contiguous memory region described
     * by {@code dst} buffer. The routine returns immediately and does not guarantee that
     * remote data is loaded and stored under the local buffer.
     */
    public void getNonBlockingImplicit(long remoteAddress, UcpRemoteKey remoteKey,
                                       ByteBuffer dst) {
        getNonBlockingImplicit(new long[] { remoteAddress },
                               new UcpRemoteKey[] { remoteKey },
                               new long[] { UcxUtils.getAddress(dst) },
                               new long[] { dst.remaining() });
    }

    /**
     * Non-blocking implicit remote memory get operation.
     * This routine initiate a load of contiguous blocks of data that is described
     * by the remote memory addresses {@code remoteAddresses} and the
     * {@code remoteKeys} "memory handles" in the local contiguous memory region described
     * by {@code dst} buffer. The routine returns immediately and does not guarantee that
     * remote data is loaded and stored under the local buffer.
     */
    public void getNonBlockingImplicit(long[] remoteAddresses, UcpRemoteKey[] remoteKeys,
                                       long[] sizes, ByteBuffer dst) {
        long acc = 0;
        long startAddress = UcxUtils.getAddress(dst);
        long[] addresses = new long[remoteAddresses.length];
        for (int i = 0; i < sizes.length; i++) {
            addresses[i] = startAddress + acc;
            acc += sizes[i];
        }
        getNonBlockingImplicit(remoteAddresses, remoteKeys, addresses, sizes);
    }

    /**
     * Non-blocking implicit remote memory get operation.
     * This routine initiate a load of contiguous blocks of data that is described
     * by the remote memory addresses {@code remoteAddresses} and the
     * {@code remoteKeys} "memory handles" in the multiple local memory regions described
     * by {@code dst} buffers. The routine returns immediately and does not guarantee that
     * remote data is loaded and stored under the local buffer.
     */
    public void getNonBlockingImplicit(long[] remoteAddresses, UcpRemoteKey[] remoteKeys,
                                       ByteBuffer[] dst) {
        getNonBlockingImplicit(remoteAddresses, remoteKeys,
            Arrays.stream(dst).mapToLong(UcxUtils::getAddress).toArray(),
            Arrays.stream(dst).mapToLong(ByteBuffer::remaining).toArray());
    }

    /**
     * Non-blocking implicit remote memory get operation.
     * This routine initiate a load of contiguous blocks of data that is described
     * by the remote memory addresses {@code remoteAddresses} and the
     * {@code remoteKeys} "memory handles" in the multiple local memory regions described
     * by {@code localAddresses} remoteAddresses.
     * The routine returns immediately and does not guarantee that
     * remote data is loaded and stored under the local buffer.
     */
    public void getNonBlockingImplicit(long[] remoteAddresses, UcpRemoteKey[] remoteKeys,
                                       long[] localAddresses, long[] sizes) {

        long[] rkeys = Arrays.stream(remoteKeys).mapToLong(UcxNativeStruct::getNativeId).toArray();
        getNonBlockingImplicitNative(getNativeId(), remoteAddresses, rkeys, localAddresses, sizes);
    }

    /**
     * Non-blocking tagged-send operations
     * This routine sends a messages that is described by the local buffer {@code sendBuffer},
     * starting of it's {@code sendBuffer.position()} and size {@code sendBuffer.remaining()}.
     * to the destination endpoint. Each message is associated with a {@code tag} value
     * that is used for message matching on the
     * {@link UcpWorker#recvTaggedNonBlocking(ByteBuffer, long, long, UcxCallback)}
     * "receiver".  The routine is non-blocking and therefore returns immediately,
     * however the actual send operation may be delayed.
     * The send operation is considered completed when  it is safe to reuse the source
     * {@code data} buffer. {@code callback} is invoked on completion of this operation.
     */
    public UcxRequest sendTaggedNonBlocking(ByteBuffer sendBuffer, long tag, UcxCallback callback) {
        if (!sendBuffer.isDirect()) {
            throw new UcxException("Send buffer must be direct.");
        }
        return sendTaggedNonBlockingNative(getNativeId(),
            UcxUtils.getAddress(sendBuffer), sendBuffer.remaining(), tag, callback);
    }

    public UcxRequest sendTaggedNonBlocking(long localAddress, long size,
                                            long tag, UcxCallback callback) {

        return sendTaggedNonBlockingNative(getNativeId(),
            localAddress, size, tag, callback);
    }


    /**
     * Non blocking send operation. Invokes
     * {@link UcpEndpoint#sendTaggedNonBlocking(ByteBuffer, long, UcxCallback)} with default 0 tag.
     */
    public UcxRequest sendTaggedNonBlocking(ByteBuffer sendBuffer, UcxCallback callback) {
        return sendTaggedNonBlocking(sendBuffer, 0, callback);
    }

    /**
     * This routine flushes all outstanding AMO and RMA communications on this endpoint.
     * All the AMO and RMA operations issued on this endpoint prior to this call
     * are completed both at the origin and at the target.
     */
    public UcxRequest flushNonBlocking(UcxCallback callback) {
        return flushNonBlockingNative(getNativeId(), callback);
    }

    private static native long createEndpointNative(UcpEndpointParams params, long workerId);

    private static native void destroyEndpointNative(long epId);

    private static native UcpRemoteKey unpackRemoteKey(long epId, long rkeyAddress);

    private static native UcxRequest putNonBlockingNative(long enpointId, long localAddress,
                                                          long size, long remoteAddr,
                                                          long ucpRkeyId, UcxCallback callback);

    private static native UcxRequest getNonBlockingNative(long enpointId, long remoteAddress,
                                                          long ucpRkeyId, long localAddress,
                                                          long size, UcxCallback callback);

    private static native void getNonBlockingImplicitNative(long enpointId, long[] remoteAddresses,
                                                            long[] ucpRkeyIds, long[] lAddresses,
                                                            long[] sizes);

    private static native UcxRequest sendTaggedNonBlockingNative(long enpointId, long localAddress,
                                                                 long size, long tag,
                                                                 UcxCallback callback);

    private static native UcxRequest flushNonBlockingNative(long enpointId, UcxCallback callback);
}
