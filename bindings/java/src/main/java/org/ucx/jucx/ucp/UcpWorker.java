/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import java.io.Closeable;
import java.nio.ByteBuffer;

import org.ucx.jucx.UcxCallback;
import org.ucx.jucx.UcxException;
import org.ucx.jucx.UcxNativeStruct;
import org.ucx.jucx.UcxRequest;

/**
 * UCP worker is an opaque object representing the communication context.  The
 * worker represents an instance of a local communication resource and progress
 * engine associated with it. Progress engine is a construct that is
 * responsible for asynchronous and independent progress of communication
 * directives. The progress engine could be implement in hardware or software.
 * The worker object abstract an instance of network resources such as a host
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
    }

    /**
     * This routine explicitly progresses all communication operations on a worker.
     * @return Non-zero if any communication was progressed, zero otherwise.
     */
    public int progress() {
        return progressWorkerNative(getNativeId());
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
     * to the {@code recvBuffer}. In order to notify the application about completion of the receive
     * operation the UCP library will invoke the call-back {@code callback} when the received
     * message is in the receive buffer and ready for application access.
     *
     * @param tagMask - bit mask that indicates the bits that are used for the matching of the
     * incoming tag against the expected tag.
     */
    public UcxRequest recvTaggedNonBlocking(ByteBuffer recvBuffer, long tag, long tagMask,
                                            UcxCallback callback) {
        if (!recvBuffer.isDirect()) {
            throw new UcxException("Recv buffer must be direct.");
        }
        return recvTaggedNonBlockingNative(getNativeId(), recvBuffer, tag, tagMask, callback);
    }

    /**
     * Non-blocking receive operation. Invokes
     * {@link UcpWorker#recvTaggedNonBlocking(ByteBuffer, long, long, UcxCallback)}
     * with default tag=0 and tagMask=0.
     */
    public UcxRequest recvTaggedNonBlocking(ByteBuffer recvBuffer, UcxCallback callback) {
        return recvTaggedNonBlocking(recvBuffer, 0, 0, callback);
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

    private static native long createWorkerNative(UcpWorkerParams params, long ucpContextId);

    private static native void releaseWorkerNative(long workerId);

    private static native ByteBuffer workerGetAddressNative(long workerId);

    private static native void releaseAddressNative(long workerId, ByteBuffer addressId);

    private static native int progressWorkerNative(long workerId);

    private static native void waitWorkerNative(long workerId);

    private static native void signalWorkerNative(long workerId);

    private static native UcxRequest recvTaggedNonBlockingNative(long workerId, ByteBuffer recvBuffer,
                                                                 long tag, long tagMask,
                                                                 UcxCallback callback);
}
