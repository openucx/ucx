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
public class Worker implements Closeable {
    public static final int     MAX_QUEUED_COMPLETIONS  = (1 << 20);
    public static final long    DEFAULT_REQUEST_ID      = -1L;

    private long            nativeId;
    private CompletionQueue compQueue;
    private byte[]          workerAddress;
    private Callback        callback;
    private int             maxCompletions;
    private boolean         closed;
    private Set<EndPoint>   endPoints;
    private int             outstandingRequests;
    private CommandType[]   commands;
    private boolean         threadSafety;
    private CompletionStatus[] compStatus;

    /**
     * Checks if multithread support was enabled. If true allows concurrent
     * access to Worker object.</br>
     * Different Workers can always be accessed concurrently by different threads.
     *
     * @return true if, and only if, multithread support is enabled
     */
    public static boolean isMultiThreadSupportEnabled() {
        return Bridge.isMultiThreadSupportEnabled();
    }

    /**
     * Creates a new Worker.
     *
     * @param cb
     *            Implementation of Worker.Callback interface
     *
     * @param maxCompletions
     *            Number of maximum queued completed events
     *
     * @param threadSafety
     *            Weather this Worker will be accessed from multiple threads.
     *            Pass true only if {@link Worker#isMultiThreadSupportEnabled()}
     *            == true
     *
     * @throws IllegalArgumentException
     *             In case cb == null or maxCompletions <= 0 or maxCompletions >
     *             MAX_QUEUED_COMPLETIONS
     *
     * @throws UnsatisfiedLinkError
     *             In case an error while loading native libraries
     *
     * @throws UnsupportedMultiThreadModeException
     *             In case threadSafety == true and
     *             {@link Worker#isMultiThreadSupportEnabled()} == false
     *
     * @throws IOException
     *             In case native Worker creation failed
     */
    public Worker(Callback cb, int maxCompletions, boolean threadSafety) throws IOException {
        if (cb == null || maxCompletions <= 0 ||
            maxCompletions > MAX_QUEUED_COMPLETIONS) {
            throw new IllegalArgumentException();
        }

        // An error occurred while loading native libraries
        if (LoadLibrary.errorMessage != null) {
            throw new UnsatisfiedLinkError(LoadLibrary.errorMessage);
        }

        if (threadSafety && !isMultiThreadSupportEnabled()) {
            throw new UnsupportedMultiThreadModeException();
        }

        this.outstandingRequests = 0;
        this.callback       = cb;
        this.maxCompletions = maxCompletions;
        this.workerAddress  = null;
        this.compQueue      = new CompletionQueue(); // Shared queue wrapper
        this.endPoints      = new HashSet<>();
        this.commands       = CommandType.values();
        this.compStatus     = CompletionStatus.values();
        this.threadSafety   = threadSafety;
        this.nativeId       = Bridge.createWorker(maxCompletions,
                                                  compQueue, this,
                                                  threadSafety);
        if (nativeId == 0) {
            throw new IOException("Failed to create Worker");
        }
        this.closed         = false;

        // align Java side shared queue operations endianness to be as
        // allocated in native (C) code, in nativeCreateWorker()
        compQueue.initCompletionQueue();
    }

    /**
     * Creates a new Worker. Same as
     * {@link Worker#Worker(Callback, int, boolean)
     * new Worker(cb, maxCompletions, false)}
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
        this(cb, maxCompletions, false);
    }

    /**
     * Create a new EndPoint object linked to this Worker.
     *
     * @param remoteAddr
     *            Address of the remote worker the new EndPoint will be
     *            connected to
     *
     * @return new EndPoint object
     *
     * @throws IOException
     *             In case native EndPoint creation fails
     */
    public EndPoint createEndPoint(byte[] remoteAddress) throws IOException {
        EndPoint endPoint = new EndPoint(this, remoteAddress);
        addEndPoint(endPoint);
        return endPoint;
    }

    /**
     * Destroy EndPoint object associated with this Worker, and release all
     * allocated native resources.
     * <p>
     * <b>Note:</b> endPoint shouldn't be used after calling this function. Memory
     * leaks may occur if this function is called before all requests are
     * completed.
     * </p>
     *
     * @param endPoint
     *            EndPoint to release
     *
     * @throws IllegalArgumentException
     *             In case passed endPoint is not associated with this Worker
     *
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

    public void progress() throws IOException {
        int completions = Bridge.progressWorker(this);
        compQueue.switchPrimaryQueue();
        executeCallback(completions);
    }

    private void executeCallback(int numOfCompletions) throws IOException {
        outstandingRequests -= numOfCompletions;

        for (int i = 0; i < numOfCompletions; i++) {
            CommandType cmd = commands[compQueue.completionBuff.getInt()];
            long requestId  = compQueue.completionBuff.getLong();
            long length     = compQueue.completionBuff.getLong();
            CompletionStatus status = compStatus[compQueue.completionBuff.getInt()];

            switch (cmd) {
            case JUCX_STREAM_SEND:
                if (requestId != DEFAULT_REQUEST_ID) {
                    callback.sendCompletionHandler(requestId, status);
                }
                break;

            case JUCX_STREAM_RECV:
                if (requestId != DEFAULT_REQUEST_ID) {
                    callback.recvCompletionHandler(requestId,
                                                   status,
                                                   (int) length);
                }
                break;

            case JUCX_INVALID:
            default:
                throw new IOException(cmd.name() + " operation failed");
            }
        }
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
     * Tells weather or not this Worker is thread safe.
     *
     * @return true if, and only if, this Worker can be accessed from multiple
     *         threads
     */
    public boolean isThreadSafe() {
        return threadSafety;
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
     * Getter for this Worker's outstandingRequests.
     *
     * @return Current outstanding requests for this Worker
     */
    public int getOutstandingRequests() {
        return outstandingRequests;
    }

    /**
     * Frees all resources associated with this Worker.</br>
     * Worker should not be used after calling this method.
     */
    @Override
    public void close() {
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

    synchronized boolean fetchAndIncOutstandingRequests() {
        if (outstandingRequests == maxCompletions) {
            return true;
        }
        ++outstandingRequests;
        return false;
    }

    synchronized void decOutstandingRequests() {
        --outstandingRequests;
    }

    /**
     * Translation from native code commands::Type enum to Java enum class
     */
    private enum CommandType {
        JUCX_INVALID,
        JUCX_STREAM_SEND,
        JUCX_STREAM_RECV,
    };


    /**
     * Translation from native code commands::CompStatus
     * enum to Java enum class
     */
    public enum CompletionStatus {
        JUCX_OK,
        JUCX_ERR,
        JUCX_ERR_CANCELED,
    }


    /**
     * Wrapper object for shared buffer between Java and native code.
     */
    class CompletionQueue {
        private ByteBuffer completionBuff   = null;
        private int primaryQueueOffset      = 0;
        private int capacity                = 0;

        private void initCompletionQueue() {
            capacity = completionBuff.capacity();
            completionBuff.order(ByteOrder.nativeOrder());
        }

        private void switchPrimaryQueue() {
            primaryQueueOffset = (primaryQueueOffset + capacity/2) % capacity;
            completionBuff.position(primaryQueueOffset);
        }
    }


    /**
     * The Callback interface must be implemented in-order to create a
     * Worker.</br>
     * Worker will invoke the implemented method whenever a request is
     * completed.
     */
    public static interface Callback {

        /**
         * Invoked whenever a send request is completed.
         *
         * @param requestId
         *            Id of completed request
         *
         * @param completionStatus
         *            Status of completed request. One of
         *            {@link CompletionStatus#JUCX_OK},
         *            {@link CompletionStatus#JUCX_ERR},
         *            {@link CompletionStatus#JUCX_ERR_CANCELED}
         */
        public void sendCompletionHandler(long requestId,
                                          CompletionStatus completionStatus);

        /**
         * Invoked whenever a receive request is completed.
         *
         * @param requestId
         *            Id of completed request
         *
         * @param completionStatus
         *            Status of completed request. One of
         *            {@link CompletionStatus#JUCX_OK},
         *            {@link CompletionStatus#JUCX_ERR},
         *            {@link CompletionStatus#JUCX_ERR_CANCELED}
         *
         * @param length
         *            length of received data (in bytes)
         */
        public void recvCompletionHandler(long requestId,
                                          CompletionStatus completionStatus,
                                          int length);
    }
}
