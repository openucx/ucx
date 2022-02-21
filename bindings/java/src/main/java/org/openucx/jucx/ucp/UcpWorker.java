/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.HashMap;

import org.openucx.jucx.*;

/**
 * UCP worker is an opaque object representing the communication context. The
 * worker represents an instance of a local communication resource and the
 * progress engine associated with it. The progress engine is a construct that
 * is responsible for asynchronous and independent progress of communication
 * directives. The progress engine could be implemented in hardware or software.
 * The worker object abstracts an instance of network resources such as a host
 * channel adapter port, network interface, or multiple resources such as
 * multiple network interfaces or communication ports. It could also represent
 * virtual communication resources that are defined across multiple devices.
 * Although the worker can represent multiple network resources, it is
 * associated with a single {@link UcpContext} "UCX application context".
 * All communication functions require a context to perform the operation on
 * the dedicated hardware resource(s) and an "endpoint" to address the
 * destination.
 *
 * <p>Worker are parallel "threading points" that an upper layer may use to
 * optimize concurrent communications.
 */
public class UcpWorker extends UcxNativeStruct implements Closeable {

    /**
     * To keep a reference to AmRecvCallback class to prevent it from GC.
     */
    private final HashMap<Integer, Object[]> amRecvHandlers = new HashMap<>();

    private long maxAmHeaderSize = 0L;

    public UcpWorker(UcpContext context, UcpWorkerParams params) {
        setNativeId(createWorkerNative(params, context.getNativeId()));
    }

    /**
     * Creates new UcpEndpoint on current worker.
     */
    public UcpEndpoint newEndpoint(UcpEndpointParams params) {
        return new UcpEndpoint(this, params);
    }

    /**
     * Creates new UcpListener on current worker.
     */
    public UcpListener newListener(UcpListenerParams params) {
        return new UcpListener(this, params);
    }

    @Override
    public void close() {
        releaseWorkerNative(getNativeId());
        setNativeId(null);
        amRecvHandlers.clear();
    }

    /**
     * Maximal allowed header size for {@link UcpEndpoint#sendAmNonBlocking} routine.
     */
    public long getMaxAmHeaderSize() {
        return maxAmHeaderSize;
    }

    /**
     * This routine installs a user defined callback to handle incoming Active
     * Messages with a specific id. This callback is called whenever an Active
     * Message that was sent from the remote peer by @ref ucp_am_send_nbx is
     * received on this worker.
     *
     * @param callback - Active Message callback. To clear the already set callback,
     *                   this value should be set to null.
     */
    public void setAmRecvHandler(int amId, UcpAmRecvCallback callback, long flags) {
        if (callback == null) {
            removeAmRecvHandler(amId);
            return;
        }
        Object[] callbackAndWorker = new Object[2];
        callbackAndWorker[0] = callback;
        callbackAndWorker[1] = this;
        amRecvHandlers.put(amId, callbackAndWorker);
        setAmRecvHandlerNative(getNativeId(), amId, callbackAndWorker, flags);
    }

    public void setAmRecvHandler(int amId, UcpAmRecvCallback callback) {
        setAmRecvHandler(amId, callback, 0L);
    }

    /**
     * Clears Active Message callback.
     */
    public void removeAmRecvHandler(int amId) {
        amRecvHandlers.remove(amId);
        setAmRecvHandlerNative(getNativeId(), amId, null, 0L);
    }

    /**
     * This routine releases data that persisted through an Active Message
     * callback because that callback returned UCS_INPROGRESS.
     */
    public void amDataRelease(long address) {
        amDataReleaseNative(getNativeId(), address);
    }

    /**
     * This routine receives a message that is described by the data descriptor
     * {@code dataDesc}, local address {@code address} and size {@code size} on a worker.
     * The routine is non-blocking and therefore returns immediately.
     * The receive operation is considered completed when the message is delivered to the buffer.
     */
    public UcpRequest recvAmDataNonBlocking(long dataDesc, long address, long size,
                                            UcxCallback callback, UcpRequestParams params) {
        return recvAmDataNonBlockingNative(getNativeId(), dataDesc, address, size, callback,
            params);
    }

    public UcpRequest recvAmDataNonBlocking(long dataDesc, long address, long size,
                                            UcxCallback callback, int memoryType) {
        return recvAmDataNonBlocking(dataDesc, address, size, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }


    /**
     * This routine explicitly progresses all communication operations on a worker.
     * @return Non-zero if any communication was progressed, zero otherwise.
     */
    public int progress() throws Exception {
        return progressWorkerNative(getNativeId());
    }

    /**
     * Blocking progress for request until it's not completed.
     */
    public void progressRequest(UcpRequest request) throws Exception {
        while (!request.isCompleted()) {
            progress();
        }
    }

    /**
     * This routine flushes all outstanding AMO and RMA communications on the
     * this worker. All the AMO and RMA operations issued on this  worker prior to this call
     * are completed both at the origin and at the target when this call returns.
     */
    public UcpRequest flushNonBlocking(UcxCallback callback) {
        return flushNonBlockingNative(getNativeId(), callback);
    }

    /**
     * This routine waits (blocking) until an event has happened, as part of the
     * wake-up mechanism.
     *
     * This function is guaranteed to return only if new communication events occur
     * on the worker. Therefore one must drain all existing events before waiting
     * on the file descriptor. This can be achieved by calling
     * {@link UcpWorker#progress()} repeatedly until it returns 0.
     */
    public void waitForEvents() {
        waitWorkerNative(getNativeId());
    }

    /**
     * This routine signals that the event has happened, as part of the wake-up
     * mechanism. This function causes a blocking call to {@link UcpWorker#waitForEvents()}
     * to return, even if no event from the underlying interfaces has taken place.
     *
     * It’s safe to use this routine from any thread, even if UCX is compiled
     * without multi-threading support and/or initialized without
     * {@link UcpWorkerParams#requestThreadSafety()}. However {@link UcpContext} has to be
     * created with {@link UcpParams#requestWakeupFeature()}.
     */
    public void signal() {
        signalWorkerNative(getNativeId());
    }

    /**
     * Non-blocking tagged-receive operation.
     * This routine receives a messages that is described by the local {@code recvBuffer}
     * buffer on the current worker. The tag value of the receive message has to match
     * the {@code tag} of sent message. The routine is a non-blocking and therefore returns
     * immediately. The receive operation is considered completed when the message is delivered
     * to the {@code recvBuffer} at position {@code recvBuffer.position()} and size
     * {@code recvBuffer.remaining()}.
     * In order to notify the application about completion of the receive
     * operation the UCP library will invoke the call-back {@code callback} when the received
     * message is in the receive buffer and ready for application access.
     *
     * @param tagMask - bit mask that indicates the bits that are used for the matching of the
     * incoming tag against the expected tag.
     */
    public UcpRequest recvTaggedNonBlocking(ByteBuffer recvBuffer, long tag, long tagMask,
                                            UcxCallback callback) {
        if (!recvBuffer.isDirect()) {
            throw new UcxException("Recv buffer must be direct.");
        }
        return recvTaggedNonBlocking(UcxUtils.getAddress(recvBuffer),
            recvBuffer.remaining(), tag, tagMask, callback);
    }

    public UcpRequest recvTaggedNonBlocking(long localAddress, long size, long tag, long tagMask,
                                            UcxCallback callback) {
        return recvTaggedNonBlocking(localAddress, size, tag, tagMask, callback, null);
    }

    public UcpRequest recvTaggedNonBlocking(long localAddress, long size, long tag, long tagMask,
                                            UcxCallback callback, int memoryType) {
        return recvTaggedNonBlocking(localAddress, size, tag, tagMask, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest recvTaggedNonBlocking(long localAddress, long size, long tag, long tagMask,
                                            UcxCallback callback, UcpRequestParams params) {
        return recvTaggedNonBlockingNative(getNativeId(), localAddress, size,
            tag, tagMask, callback, params);
    }

    /**
     * Non-blocking receive operation. Invokes
     * {@link UcpWorker#recvTaggedNonBlocking(ByteBuffer, long, long, UcxCallback)}
     * with default tag=0 and tagMask=0.
     */
    public UcpRequest recvTaggedNonBlocking(ByteBuffer recvBuffer, UcxCallback callback) {
        return recvTaggedNonBlocking(recvBuffer, 0, 0, callback);
    }

    public UcpRequest recvTaggedNonBlocking(long[] localAddresses, long[] sizes,
                                            long tag, long tagMask,
                                            UcxCallback callback) {

        return recvTaggedNonBlocking(localAddresses, sizes, tag, tagMask, callback,
            null);
    }

    public UcpRequest recvTaggedNonBlocking(long[] localAddresses, long[] sizes,
                                            long tag, long tagMask,
                                            UcxCallback callback, UcpRequestParams params) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return recvTaggedIovNonBlockingNative(getNativeId(), localAddresses, sizes, tag,
            tagMask, callback, params);
    }

    public UcpRequest recvTaggedNonBlocking(long[] localAddresses, long[] sizes,
                                            long tag, long tagMask,
                                            UcxCallback callback, int memoryType) {
        UcxParams.checkArraySizes(localAddresses, sizes);

        return recvTaggedNonBlocking(localAddresses, sizes, tag, tagMask, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    /**
     * Non-blocking probe and return a message.
     * This routine probes (checks) if a messages described by the {@code tag} and
     * {@code tagMask} was received (fully or partially) on the worker. The tag
     * value of the received message has to match the {@code tag} and {@code tagMask}
     * values, where the {@code tagMask} indicates what bits of the tag have to be
     * matched. The function returns immediately and if the message is matched it
     * returns a handle for the message.
     *
     * This function does not advance the communication state of the network.
     * If this routine is used in busy-poll mode, need to make sure
     * {@link UcpWorker#progress()} is called periodically to extract messages from the transport.
     *
     * @param remove - The flag indicates if the matched message has to be removed from UCP library.
     *                 If true, the message handle is removed from the UCP library
     *                 and the application is responsible to call
     *                 {@link UcpWorker#recvTaggedMessageNonBlocking(long, long, UcpTagMessage,
     *                 UcxCallback)} in order to receive the data and release the resources
     *                 associated with the message handle.
     *                 If false, the return value is merely an indication to whether a matching
     *                 message is present, and it cannot be used in any other way,
     *                 and in particular it cannot be passed to
     *                 {@link UcpWorker#recvTaggedMessageNonBlocking(long, long, UcpTagMessage,
     *                 UcxCallback)}
     * @return  NULL                      - No match found.
     *          Message handle (not NULL) - If message is matched the message handle is returned.
     */
    public UcpTagMessage tagProbeNonBlocking(long tag, long tagMask, boolean remove) {
        return tagProbeNonBlockingNative(getNativeId(), tag, tagMask, remove);
    }

    /**
     * Non-blocking receive operation for a probed message.
     * This routine receives a messages that is described by the local {@code address},
     * {@code size}, and a {@code message} handle. The {@code message} handle can be obtain
     * by calling the {@link UcpWorker#tagProbeNonBlocking(long, long, boolean)}. This routine
     * is a non-blocking and therefore returns immediately. The receive operation is considered
     * completed when the message is delivered to the buffer, described by {@code address}
     * and {@code size}.
     * In order to notify the application about completion of the receive operation
     * the UCP library will invoke the call-back {@code callback} when the received message
     * is in the receive buffer and ready for application access.
     * If the receive operation cannot be stated the routine returns an error.
     */
    public UcpRequest recvTaggedMessageNonBlocking(long address, long size, UcpTagMessage message,
                                                   UcxCallback callback, UcpRequestParams params) {
        return recvTaggedMessageNonBlockingNative(getNativeId(), address, size,
            message.getNativeId(), callback, params);
    }

    public UcpRequest recvTaggedMessageNonBlocking(long address, long size, UcpTagMessage message,
                                                   UcxCallback callback, int memoryType) {
        return recvTaggedMessageNonBlocking(address, size, message, callback,
            new UcpRequestParams().setMemoryType(memoryType));
    }

    public UcpRequest recvTaggedMessageNonBlocking(long address, long size, UcpTagMessage message,
                                                   UcxCallback callback) {
        return recvTaggedMessageNonBlocking(address, size, message, callback, null);
    }

    public UcpRequest recvTaggedMessageNonBlocking(ByteBuffer buffer, UcpTagMessage message,
                                                   UcxCallback callback) {
        return recvTaggedMessageNonBlocking(UcxUtils.getAddress(buffer), buffer.remaining(),
            message, callback);
    }


    /**
     * This routine tries to cancels an outstanding communication request. After
     * calling this routine, the request will be in completed or canceled (but
     * not both) state regardless of the status of the target endpoint associated
     * with the communication request. If the request is completed successfully,
     * the "send" or the "receive" completion callbacks (based on the type of the request) will be
     * called with the status argument of the callback set to UCS_OK, and in a
     * case it is canceled the status argument is set to UCS_ERR_CANCELED.
     */
    public void cancelRequest(UcpRequest request) {
        if (request.getNativeId() == null) {
            throw new UcxException("Request is not valid");
        }
        cancelRequestNative(getNativeId(), request.getNativeId());
    }

    /**
     * This routine returns the address of the worker object. This address can be
     * passed to remote instances of the UCP library in order to connect to this
     * worker. Ucp worker address - is an opaque object that is used as an
     * identifier for a {@link UcpWorker} instance.
     */
    public ByteBuffer getAddress() {
        ByteBuffer nativeUcpAddress = workerGetAddressNative(getNativeId());
        // 1. Allocating java native ByteBuffer (managed by java's reference count cleaner).
        ByteBuffer result = ByteBuffer.allocateDirect(nativeUcpAddress.capacity());
        // 2. Copy content of native ucp address to java's buffer.
        result.put(nativeUcpAddress);
        result.clear();
        // 3. Release an address of the worker object. Memory allocated in JNI must be freed by JNI.
        releaseAddressNative(getNativeId(), nativeUcpAddress);
        return result;
    }

    private native long createWorkerNative(UcpWorkerParams params, long ucpContextId);

    private static native void releaseWorkerNative(long workerId);

    private static native ByteBuffer workerGetAddressNative(long workerId);

    private static native void releaseAddressNative(long workerId, ByteBuffer addressId);

    private static native int progressWorkerNative(long workerId) throws Exception;

    private static native UcpRequest flushNonBlockingNative(long workerId, UcxCallback callback);

    private static native void waitWorkerNative(long workerId);

    private static native void signalWorkerNative(long workerId);

    private static native void setAmRecvHandlerNative(long workerId, int amId,
                                                      Object[] callbackAndWorker,
                                                      long flags);

    private static native UcpRequest recvAmDataNonBlockingNative(long workerId, long dataDesc,
                                                                 long address, long size,
                                                                 UcxCallback callback,
                                                                 UcpRequestParams params);

    private static native void amDataReleaseNative(long workerId, long dataAddress);

    private static native UcpRequest recvTaggedNonBlockingNative(long workerId, long localAddress,
                                                                 long size, long tag, long tagMask,
                                                                 UcxCallback callback,
                                                                 UcpRequestParams params);

    private static native UcpRequest recvTaggedIovNonBlockingNative(long workerId,
                                                                    long[] localAddresses,
                                                                    long[] sizes,
                                                                    long tag, long tagMask,
                                                                    UcxCallback callback,
                                                                    UcpRequestParams params);

    private static native UcpTagMessage tagProbeNonBlockingNative(long workerId, long tag,
                                                                  long tagMask, boolean remove);

    private static native UcpRequest recvTaggedMessageNonBlockingNative(long workerId, long address,
                                                                        long size, long tagMsgId,
                                                                        UcxCallback callback,
                                                                        UcpRequestParams params);

    private static native void cancelRequestNative(long workerId, long requestId);
}
