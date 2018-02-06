/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.Set;

/**
 * Worker is the object representing a local communication resource such as a
 * network interface or host channel adapter port.
 */
@SuppressWarnings("unused")
public class Worker implements Closeable {
    public static final int MAX_QUEUED_COMPLETIONS = (1 << 20);

    private long            nativeId;
    private CompletionQueue compQueue;
    private byte[]          workerAddress;
    private Callback        callback;
    private int             maxCompletions;
    private boolean         closed;
    private Set<EndPoint>   endPoints;

    /**
     * Creates a new Worker.
     *
     * @param cb
     *            Implementation of Worker.Callback interface
     *
     * @param maxCompletions
     *            Number of maximum queued completed events
     *
     * @throws IllegalArgumentException
     *             In case cb == null or maxCompletions <= 0 or maxCompletions >
     *             MAX_QUEUED_COMPLETIONS
     *
     * @throws UnsatisfiedLinkError
     *             In case an error while loading native libraries
     *
     * @throws IOException
     *             In case native Worker creation failed
     */
    public Worker(Callback cb, int maxCompletions) throws IOException {
        if (cb == null || maxCompletions <= 0 ||
            maxCompletions > MAX_QUEUED_COMPLETIONS) {
            throw new IllegalArgumentException();
        }

        // An error occurred while loading native libraries
        if (LoadLibrary.errorMessage != null) {
            throw new UnsatisfiedLinkError(LoadLibrary.errorMessage);
        }

        this.callback       = cb;
        this.maxCompletions = maxCompletions;
        this.workerAddress  = null;
        this.compQueue      = new CompletionQueue(); // Shared queue wrapper
        this.endPoints      = new HashSet<>();
        this.nativeId       = Bridge.createWorker(maxCompletions, compQueue, this);
        if (nativeId == 0) {
            throw new IOException("Failed to create Worker");
        }
        this.closed         = false;

        // align Java side shared queue operations endianness to be as
        // allocated in native (C) code, in nativeCreateWorker()
        this.compQueue.setEndianness();
    }


    /**
     * Create a new EndPoint object linked to this Worker.
     *
     * @param workerAddress
     *            Address of the remote worker the new EndPoint will be
     *            connected to
     *
     * @return new EndPoint object
     *
     * @throws IOException
     *             In case native EndPoint creation fails
     */
    public EndPoint createEndPoint(byte[] workerAddress) throws IOException {
        EndPoint endPoint = new EndPoint(this, workerAddress);
        addEndPoint(endPoint);
        return endPoint;
    }

    /**
     * Destroy EndPoint object associated with this Worker, and release all
     * allocated native resources.
     * <p>
     * Note: endPoint shouldn't be used after calling this function.
     * </p>
     *
     * @param endPoint
     *            EndPoint to release
     *
     * @throws IllegalArgumentException
     *             In case passed endPoint is not associated with this Worker
     */
    public void destroyEndPoint(EndPoint endPoint) {
        if (!endPoints.contains(endPoint)) {
            throw new IllegalArgumentException("endPoint not associated with this Worker");
        }

        endPoint.close();
        removeEndPoint(endPoint);
    }

    private void addEndPoint(EndPoint ep) {
        endPoints.add(ep);
    }

    private void removeEndPoint(EndPoint ep) {
        endPoints.remove(ep);
    }

    /**
     * Getter for native pointer as long.
     *
     * @return long integer representing native pointer
     */
    long getNativeId() {
        return nativeId;
    }

    /**
     * Getter for workerAddress as a byte array.
     *
     * @return clone of address (for safety reasons)
     */
    public byte[] getWorkerAddress() {
        return workerAddress.clone();
    }

    /**
     * Getter for this Worker's maxCompletions (maximum outstanding requests).
     *
     * @return Maximum number of possible outstanding requests for this Worker
     */
    public int getMaxCompletions() {
        return maxCompletions;
    }

    /**
     * Frees all resources associated with this Worker.</br>
     * Worker should not be used after calling this method.
     */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }

        for (EndPoint ep : endPoints) {
            destroyEndPoint(ep);
        }
        closed = true;
        Bridge.destroyWorker(this);
    }

    /**
     * Called when object is garbage collected. Frees native allocated
     * resources.
     */
    @Override
    protected void finalize() throws Throwable {
        close();
    }


    /**
     * Wrapper object for shared buffer between Java and native code.
     */
    class CompletionQueue {
        ByteBuffer completionBuff = null;

        private void setEndianness() {
            completionBuff.order(ByteOrder.nativeOrder());
        }
    }


    /**
     * The Callback interface must be implemented in-order to create a
     * Worker.</br>
     * Worker will invoke the implemented method whenever a request is
     * completed.
     */
    public static interface Callback {
        // Handlers to implement will be added when data path is added
    }
}
