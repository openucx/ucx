/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.*;

import java.io.Closeable;
import java.nio.ByteBuffer;

public class UcpEndpoint extends UcxNativeStruct implements Closeable {
    private String paramsString;
    // Keep a reference to errorHandler to prevent it from GC and have valid ref
    // from JNI error handler.
    private UcpEndpointErrorHandler errorHandler;

    /**
     * To construct reply endpoint for Active Messages from JNI.
     */
    private UcpEndpoint(long nativeId) {
        setNativeId(nativeId);
    }

    @Override
    public String toString() {
        return "UcpEndpoint(id=" + getNativeId() + ", " + paramsString + ")";
    }

    public UcpEndpoint(UcpWorker worker, UcpEndpointParams params) {
        // For backward compatibility and better error tracking always set ep error handler.
        if (params.errorHandler == null) {
            params.setErrorHandler((ep, status, errorMsg) -> {
                throw new UcxException("Endpoint " + ep.toString() +
                    " error: " + errorMsg);
            });
        }
        this.errorHandler = params.errorHandler;
        this.paramsString = params.toString();
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
    public UcpRequest putNonBlocking(ByteBuffer src, long remoteAddress, UcpRemoteKey remoteKey,
                                     UcxCallback callback) {

        checkRemoteAccessParams(src, remoteKey);

        return putNonBlocking(UcxUtils.getAddress(src), src.remaining(), remoteAddress,
            remoteKey, callback);
    }

    public UcpRequest putNonBlocking(long localAddress, long size,
                                     long remoteAddress, UcpRemoteKey remoteKey,
                                     UcxCallback callback) {
        return putNonBlocking(localAddress, size, remoteAddress, remoteKey, callback,
            null);
    }

    public UcpRequest putNonBlocking(long localAddress, long size,
                                     long remoteAddress, UcpRemoteKey remoteKey,
                                     UcxCallback callback, int memoryType) {
        return putNonBlocking(localAddress, size, remoteAddress, remoteKey, callback,
                              new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest putNonBlocking(long localAddress, long size,
                                     long remoteAddress, UcpRemoteKey remoteKey,
                                     UcxCallback callback, UcpRequestParams params) {
        return putNonBlockingNative(getNativeId(), localAddress,
                                    size, remoteAddress, remoteKey.getNativeId(),
                                    callback, params);
    }

    /**
     * This routine initiates a storage of contiguous block of data that is
     * described by the local {@code buffer} in the remote contiguous memory
     * region described by {@code remoteAddress} and the {@code remoteKey}
     * "memory handle". The routine returns immediately and does not
     * guarantee re-usability of the source {@code src} buffer.
     */
    public void putNonBlockingImplicit(ByteBuffer src, long remoteAddress,
                                       UcpRemoteKey remoteKey) {
        checkRemoteAccessParams(src, remoteKey);

        putNonBlockingImplicit(UcxUtils.getAddress(src), src.remaining(), remoteAddress,
            remoteKey);
    }

    /**
     * This routine initiates a storage of contiguous block of data that is
     * described by the local {@code localAddress} in the remote contiguous memory
     * region described by {@code remoteAddress} and the {@code remoteKey}
     * "memory handle". The routine returns immediately and does not
     * guarantee re-usability of the source {@code localAddress} address.
     */
    public void putNonBlockingImplicit(long localAddress, long size,
                                       long remoteAddress, UcpRemoteKey remoteKey) {
        putNonBlockingImplicitNative(getNativeId(), localAddress, size, remoteAddress,
            remoteKey.getNativeId());
    }

    /**
     * Non-blocking remote memory get operation.
     * This routine initiates a load of a contiguous block of data that is
     * described by the remote memory address {@code remoteAddress} and the
     * {@code remoteKey} "memory handle". The routine returns immediately and <strong>does</strong>
     * not guarantee that remote data is loaded and stored under the local {@code dst} buffer
     * starting of it's {@code dst.position()} and size {@code dst.remaining()}.
     * {@code callback} is invoked on completion of this operation.
     * @return {@link UcpRequest} object that can be monitored for completion.
     */
    public UcpRequest getNonBlocking(long remoteAddress, UcpRemoteKey remoteKey,
                                     ByteBuffer dst, UcxCallback callback) {

        checkRemoteAccessParams(dst, remoteKey);

        return getNonBlocking(remoteAddress, remoteKey, UcxUtils.getAddress(dst),
            dst.remaining(), callback);
    }

    public UcpRequest getNonBlocking(long remoteAddress, UcpRemoteKey remoteKey,
                                     long localAddress, long size, UcxCallback callback) {

        return getNonBlocking(remoteAddress, remoteKey, localAddress, size, callback,
            null);
    }

    public UcpRequest getNonBlocking(long remoteAddress, UcpRemoteKey remoteKey,
                                     long localAddress, long size, UcxCallback callback,
                                     int memoryType) {

        return getNonBlocking(remoteAddress, remoteKey, localAddress, size, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest getNonBlocking(long remoteAddress, UcpRemoteKey remoteKey,
                                     long localAddress, long size, UcxCallback callback,
                                     UcpRequestParams params) {

        return getNonBlockingNative(getNativeId(), remoteAddress, remoteKey.getNativeId(),
            localAddress, size, callback, params);
    }

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
        checkRemoteAccessParams(dst, remoteKey);

        getNonBlockingImplicit(remoteAddress, remoteKey, UcxUtils.getAddress(dst),
            dst.remaining());
    }

    /**
     * Non-blocking implicit remote memory get operation.
     * This routine initiate a load of contiguous block of data that is described
     * by the remote memory address {@code remoteAddress} and the
     * {@code remoteKey} "memory handle" in the local contiguous memory region described
     * by {@code localAddress} the local address. The routine returns immediately
     * and does not guarantee that remote data is loaded and stored under the local buffer.
     */
    public void getNonBlockingImplicit(long remoteAddress, UcpRemoteKey remoteKey,
                                       long localAddress, long size) {
        getNonBlockingImplicitNative(getNativeId(), remoteAddress, remoteKey.getNativeId(),
              localAddress, size);
    }

    /**
     * Non-blocking tagged-send operations
     * This routine sends a messages that is described by the local buffer {@code sendBuffer},
     * starting of it's {@code sendBuffer.position()} and size {@code sendBuffer.remaining()}.
     * to the destination endpoint. Each message is associated with a {@code tag} value
     * that is used for message matching on the
     * {@link UcpWorker#recvTaggedNonBlocking(ByteBuffer, long, long, UcxCallback)}
     * "receiver". The routine is non-blocking and therefore returns immediately,
     * however the actual send operation may be delayed.
     * The send operation is considered completed when  it is safe to reuse the source
     * {@code data} buffer. {@code callback} is invoked on completion of this operation.
     */
    public UcpRequest sendTaggedNonBlocking(ByteBuffer sendBuffer, long tag, UcxCallback callback) {
        if (!sendBuffer.isDirect()) {
            throw new UcxException("Send buffer must be direct.");
        }
        return sendTaggedNonBlocking(UcxUtils.getAddress(sendBuffer),
          sendBuffer.remaining(), tag, callback);
    }

    public UcpRequest sendTaggedNonBlocking(long localAddress, long size,
                                            long tag, UcxCallback callback) {
        return sendTaggedNonBlocking(localAddress, size, tag, callback, null);
    }

    public UcpRequest sendTaggedNonBlocking(long localAddress, long size,
                                            long tag, UcxCallback callback, int memoryType) {
        return sendTaggedNonBlocking(localAddress, size, tag, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest sendTaggedNonBlocking(long localAddress, long size,
                                            long tag, UcxCallback callback,
                                            UcpRequestParams params) {
        return sendTaggedNonBlockingNative(getNativeId(), localAddress, size, tag,
            callback, params);
    }

    /**
     * Non blocking send operation. Invokes
     * {@link UcpEndpoint#sendTaggedNonBlocking(ByteBuffer, long, UcxCallback)} with default 0 tag.
     */
    public UcpRequest sendTaggedNonBlocking(ByteBuffer sendBuffer, UcxCallback callback) {
        return sendTaggedNonBlocking(sendBuffer, 0, callback);
    }

    /**
     * Iov version of non blocking send operation
     */

    public UcpRequest sendTaggedNonBlocking(long[] localAddresses, long[] sizes,
                                            long tag, UcxCallback callback,
                                            UcpRequestParams params) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return sendTaggedIovNonBlockingNative(getNativeId(), localAddresses, sizes,
            tag, callback, params);
    }

    public UcpRequest sendTaggedNonBlocking(long[] localAddresses, long[] sizes,
                                            long tag, UcxCallback callback, int memoryType) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return sendTaggedNonBlocking(localAddresses, sizes, tag, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest sendTaggedNonBlocking(long[] localAddresses, long[] sizes,
                                            long tag, UcxCallback callback) {

        return sendTaggedNonBlocking(localAddresses, sizes, tag, callback, null);
    }

    /**
     * This routine sends data that is described by the local address to the destination endpoint.
     * The routine is non-blocking and therefore returns immediately, however the actual send
     * operation may be delayed. The send operation is considered completed when it is safe
     * to reuse the source buffer. The UCP library will schedule invocation of the call-back upon
     * completion of the send operation.
     */
    public UcpRequest sendStreamNonBlocking(long localAddress, long size, UcxCallback callback) {
        return sendStreamNonBlocking(localAddress, size, callback, null);
    }

    public UcpRequest sendStreamNonBlocking(long localAddress, long size, UcxCallback callback,
                                            UcpRequestParams params) {
        return sendStreamNonBlockingNative(getNativeId(), localAddress, size, callback, params);
    }

    public UcpRequest sendStreamNonBlocking(long localAddress, long size, UcxCallback callback,
                                            int memoryType) {
        return sendStreamNonBlocking(localAddress, size, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest sendStreamNonBlocking(long[] localAddresses, long[] sizes,
                                            UcxCallback callback) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return sendStreamNonBlocking(localAddresses, sizes, callback, null);
    }

    public UcpRequest sendStreamNonBlocking(long[] localAddresses, long[] sizes,
                                            UcxCallback callback, UcpRequestParams params) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return sendStreamIovNonBlockingNative(getNativeId(), localAddresses, sizes, callback,
            params);
    }

    public UcpRequest sendStreamNonBlocking(long[] localAddresses, long[] sizes,
                                            UcxCallback callback, int memoryType) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return sendStreamNonBlocking(localAddresses, sizes, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest sendStreamNonBlocking(ByteBuffer buffer, UcxCallback callback) {
        return sendStreamNonBlocking(UcxUtils.getAddress(buffer), buffer.remaining(), callback);
    }

    /**
     * This routine receives data that is described by the local address and a size on the endpoint.
     * The routine is non-blocking and therefore returns immediately. The receive operation is
     * considered complete when the message is delivered to the buffer.
     * In order to notify the application about completion of a scheduled receive operation,
     * the UCP library will invoke the call-back when data is in the receive buffer
     * and ready for application access.
     */
    public UcpRequest recvStreamNonBlocking(long localAddress, long size, long flags,
                                            UcxCallback callback, UcpRequestParams params) {
        return recvStreamNonBlockingNative(getNativeId(), localAddress, size, flags, callback,
            params);
    }

    public UcpRequest recvStreamNonBlocking(long localAddress, long size, long flags,
                                            UcxCallback callback, int memoryType) {
        return recvStreamNonBlocking(localAddress, size, flags, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest recvStreamNonBlocking(long localAddress, long size, long flags,
                                            UcxCallback callback) {
        return recvStreamNonBlocking(localAddress, size, flags, callback, null);
    }

    public UcpRequest recvStreamNonBlocking(long[] localAddresses, long[] sizes, long flags,
                                            UcxCallback callback, UcpRequestParams params) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return recvStreamIovNonBlockingNative(getNativeId(), localAddresses, sizes, flags,
            callback, params);
    }

    public UcpRequest recvStreamNonBlocking(long[] localAddresses, long[] sizes, long flags,
                                            UcxCallback callback, int memoryType) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return recvStreamNonBlocking(localAddresses, sizes, flags, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest recvStreamNonBlocking(long[] localAddresses, long[] sizes, long flags,
                                            UcxCallback callback) {

        return recvStreamNonBlocking(localAddresses, sizes, flags, callback, null);
    }

    public UcpRequest recvStreamNonBlocking(ByteBuffer buffer, long flags, UcxCallback callback) {
        return recvStreamNonBlocking(UcxUtils.getAddress(buffer), buffer.remaining(), flags,
            callback);
    }

    /**
     * Send Active Message.
     * @param activeMessageId - Active Message id. Specifies which callback registered by
     *                          {@link UcpWorker#setAmRecvHandler(int, UcpAmRecvCallback)} to run.
     * @param headerAddress   - User defined Active Message header. NULL value is
     *                          allowed if no header needed. In this case
     * @param headerLength    - Active message header length in bytes.
     * @param dataAddress     - Pointer to the data to be sent to the target node
     *                          of the Active Message.
     * @param dataLength      - Data length size in bytes
     * @param callback        - Callback to call on a completion.
     */
    public UcpRequest sendAmNonBlocking(int activeMessageId, long headerAddress, long headerLength,
                                        long dataAddress, long dataLength, long flags,
                                        UcxCallback callback, UcpRequestParams params) {
        return sendAmNonBlockingNative(getNativeId(), activeMessageId,
            headerAddress, headerLength, dataAddress, dataLength, flags, callback, params);
    }

    public UcpRequest sendAmNonBlocking(int activeMessageId, long headerAddress, long headerLength,
                                        long dataAddress, long dataLength, long flags,
                                        UcxCallback callback, int memoryType) {
        return sendAmNonBlocking(activeMessageId, headerAddress, headerLength, dataAddress,
            dataLength, flags, callback, new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest sendAmNonBlocking(int activeMessageId, long headerAddress, long headerLength,
                                        long dataAddress, long dataLength, long flags,
                                        UcxCallback callback) {
        return sendAmNonBlocking(activeMessageId, headerAddress, headerLength,
            dataAddress, dataLength, flags, callback, null);
    }

    /**
     * This routine flushes all outstanding AMO and RMA communications on this endpoint.
     * All the AMO and RMA operations issued on this endpoint prior to this call
     * are completed both at the origin and at the target.
     */
    public UcpRequest flushNonBlocking(UcxCallback callback) {
        return flushNonBlockingNative(getNativeId(), callback);
    }

    /**
     * Releases the endpoint without any confirmation from the peer. All
     * outstanding requests will be completed with UCS_ERR_CANCELED error.
     * This mode may cause transport level errors on remote side, so it requires set
     * {@link UcpEndpointParams#setPeerErrorHandlingMode()} for all endpoints created on
     * both (local and remote) sides to avoid undefined behavior.
     */
    public UcpRequest closeNonBlockingForce() {
        return closeNonBlockingNative(getNativeId(), UcpConstants.UCP_EP_CLOSE_FLAG_FORCE);
    }

    /**
     * Releases the endpoint by scheduling flushes on all outstanding operations.
     */
    public UcpRequest closeNonBlockingFlush() {
        return closeNonBlockingNative(getNativeId(), 0);
    }

    private native long createEndpointNative(UcpEndpointParams params, long workerId);

    private static native void destroyEndpointNative(long epId);

    private static native UcpRemoteKey unpackRemoteKey(long epId, long rkeyAddress);

    private static native UcpRequest putNonBlockingNative(long enpointId, long localAddress,
                                                          long size, long remoteAddr,
                                                          long ucpRkeyId, UcxCallback callback,
                                                          UcpRequestParams params);

    private static native void putNonBlockingImplicitNative(long enpointId, long localAddress,
                                                            long size, long remoteAddr,
                                                            long ucpRkeyId);

    private static native UcpRequest getNonBlockingNative(long enpointId, long remoteAddress,
                                                          long ucpRkeyId, long localAddress,
                                                          long size, UcxCallback callback,
                                                          UcpRequestParams params);

    private static native void getNonBlockingImplicitNative(long enpointId, long remoteAddress,
                                                            long ucpRkeyId, long localAddress,
                                                            long size);

    private static native UcpRequest sendTaggedNonBlockingNative(long enpointId, long localAddress,
                                                                 long size, long tag,
                                                                 UcxCallback callback,
                                                                 UcpRequestParams params);

    private static native UcpRequest sendTaggedIovNonBlockingNative(long enpointId,
                                                                    long[] localAddresses,
                                                                    long[] sizes, long tag,
                                                                    UcxCallback callback,
                                                                    UcpRequestParams params);

    private static native UcpRequest sendStreamNonBlockingNative(long enpointId, long localAddress,
                                                                 long size, UcxCallback callback,
                                                                 UcpRequestParams params);

    private static native UcpRequest sendStreamIovNonBlockingNative(long enpointId,
                                                                    long[] localAddresses,
                                                                    long[] sizes,
                                                                    UcxCallback callback,
                                                                    UcpRequestParams params);

    private static native UcpRequest recvStreamNonBlockingNative(long enpointId, long localAddress,
                                                                 long size, long flags,
                                                                 UcxCallback callback,
                                                                 UcpRequestParams params);

    private static native UcpRequest recvStreamIovNonBlockingNative(long enpointId,
                                                                    long[] localAddresses,
                                                                    long[] sizes, long flags,
                                                                    UcxCallback callback,
                                                                    UcpRequestParams params);

    private static native UcpRequest sendAmNonBlockingNative(long enpointId, int activeMessageId,
                                                             long headerAddress, long headerLength,
                                                             long dataAddress, long dataLength,
                                                             long flags, UcxCallback callback,
                                                             UcpRequestParams params);

    private static native UcpRequest flushNonBlockingNative(long enpointId, UcxCallback callback);

    private static native UcpRequest closeNonBlockingNative(long endpointId, int mode);
}
