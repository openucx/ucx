/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Worker is the object representing a local communication resource such as a
 * network interface or host channel adapter port.
 */
@SuppressWarnings("unused")
public class Worker implements Closeable {
    public static final int MAX_QUEUED_EVENTS = (1 << 20);

    private long            nativeId;
    private CompletionQueue compQueue;
    private byte[]          workerAddress;
    private Callback        callback;
    private int             maxEvents;
    private boolean         closed;

    /**
     * Creates a new Worker.
     *
     * @param cb
     *            Implementation of Worker.Callback interface
     *
     * @param maxEvents
     *            Number of maximum queued completed events
     *
     * @throws IllegalArgumentException
     *             In case cb == null or maxEvents <= 0 or
     *             maxEvents > MAX_QUEUED_EVENTS
     *
     * @throws UnsatisfiedLinkError
     *             In case an error while loading native libraries
     *
     * @throws IOException
     *             In case native Worker creation failed
     */
    public Worker(Callback cb, int maxEvents) throws IOException {
        if (cb == null || maxEvents <= 0 || maxEvents > MAX_QUEUED_EVENTS) {
            throw new IllegalArgumentException();
        }

        // An error occurred while loading native libraries
        if (LoadLibrary.errorMessage != null) {
            throw new UnsatisfiedLinkError(LoadLibrary.errorMessage);
        }

        this.callback       = cb;
        this.closed         = false;
        this.maxEvents      = maxEvents;
        this.workerAddress  = null;
        this.compQueue      = new CompletionQueue(); // Shared queue wrapper
        this.nativeId       = Bridge.createWorker(maxEvents, compQueue, this);
        if (nativeId == 0) {
            throw new IOException("Failed to create Worker");
        }

        // Allign Java side shared queue operations endianness to be as
        // allocated in native (C) code, in nativeCreateWorker()
        this.compQueue.setEndianness();
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
     * Frees all resources associated with this Worker.</br>
     * Worker should not be used after calling this method.
     */
    @Override
    public void close() {
        closed = true;
        Bridge.releaseWorker(this);
    }

    /**
     * Called when object is garbage collected. Frees native allocated
     * resources.
     */
    @Override
    protected void finalize() throws Throwable {
        if (!closed) {
            close();
        }
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
