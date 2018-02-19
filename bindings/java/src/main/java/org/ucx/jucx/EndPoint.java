/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.ucx.jucx.Worker.Callback;

public class EndPoint {
	private long       nativeId;
	private Worker     localWorker;
	private byte[]     remoteWorkerAddress;
	private boolean    closed;

    /**
     * Creates a new EndPoint object
     *
     * @param worker
     *            (local) Worker object this EndPoint is associated with
     *
     * @param remoteAddress
     *            Remote Worker's address this EndPoint will connect to
     */
    EndPoint(Worker worker, byte[] remoteAddress) throws IOException {
        localWorker         = worker;
        remoteWorkerAddress = remoteAddress;
        closed              = false;
        nativeId            = Bridge.createEndPoint(worker, remoteAddress);
        if (nativeId == 0) {
            throw new IOException("Failed to create EndPoint");
        }
    }

    /**
     * EndPoint posts an asynchronous send request for the data in buffer. The
     * completed event's ID will be requestId.</br>
     * {@link ByteBuffer#remaining() buffer.remaining()} bytes will be sent, but
     * {@link ByteBuffer buffer}'s position will not be updated.
     *
     * @param buffer
     *            Outgoing data to be sent
     *
     * @param requestId
     *            Send request ID
     *
     * @throws ClosedEndPointException
     *             If this EndPoint is closed
     *
     * @throws IllegalArgumentException
     *             If {@link ByteBuffer#isDirect() == false}
     *
     * @throws IllegalStateException
     *             If {@link Worker#getOutstandingRequests()} ==
     *             {@link Worker#getMaxCompletions()}
     *
     * @throws IOException
     *             if send request failed
     */
    public void streamSend(ByteBuffer buffer, long requestId) throws IOException {
        if (!buffer.isDirect()) {
            throw new IllegalArgumentException("Support only for Direct ByteBuffer");
        }
        // TODO: future devel - hold private DirectBB pool for non-directBB
        // operations

        if (localWorker.fetchAndIncOutstandingRequests()) {
            throw new IllegalStateException("Number of requests exceeds limit");
        }

        boolean error = Bridge.streamSend(this, buffer, requestId);
        if (error) {
            localWorker.decOutstandingRequests();
            throw new IOException("Failed to send data");
        }
    }

    /**
     * EndPoint posts an asynchronous receive request. The received data will
     * be filled to buffer upon completion.</br>
     * {@link ByteBuffer buffer}'s position will not be updated. Length of
     * received data will be given as an argument to
     * {@link Callback#recvCompletionHandler(long, org.ucx.jucx.Worker.CompletionStatus, int)}
     *
     * @param buffer
     *            ByteBuffer to be filled with incoming data.
     *
     * @param requestId
     *            Receive request ID
     *
     * @throws ClosedEndPointException
     *             If this EndPoint is closed
     *
     * @throws IllegalArgumentException
     *             If {@link ByteBuffer#isDirect() == false}
     *
     * @throws IllegalStateException
     *             If {@link Worker#getOutstandingRequests()} ==
     *             {@link Worker#getMaxCompletions()}
     *
     * @throws IOException
     *             if receive request failed
     */
    public void streamRecv(ByteBuffer buffer, long requestId) throws IOException {
        if (!buffer.isDirect()) {
            throw new IllegalArgumentException("Support only for Direct ByteBuffer");
        }
        // TODO: future devel - hold private DirectBB pool for non-directBB
        // operations

        if (buffer.isReadOnly()) {
            throw new IllegalArgumentException("Can't fill buffer with data:" +
                                               " buffer is Read-Only");
        }

        if (localWorker.fetchAndIncOutstandingRequests()) {
            throw new IllegalStateException("Number of requests exceeds limit");
        }

        boolean error = Bridge.streamRecv(this, buffer, requestId);
        if (error) {
            localWorker.decOutstandingRequests();
            throw new IOException("Failed to recv data");
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
     * Getter for remoteWorkerAddress as a byte array.
     *
     * @return clone of address (for safety reasons)
     */
    public byte[] getWorkerAddress() {
        return remoteWorkerAddress.clone();
    }

    /**
     * Getter for local worker
     *
     * @return Worker object
     */
    public Worker getWorker() {
        return localWorker;
    }

    /**
     * Called when object is garbage collected. Frees native allocated
     * resources.
     */
    @Override
    public void finalize() {
        close();
    }

    synchronized void close() {
        if (!closed) {
            closed = true;
            Bridge.destroyEndPoint(this);
        }
    }

    @Override
    public int hashCode() {
        return Long.hashCode(nativeId);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }

        EndPoint other = (EndPoint) obj;

        return nativeId == other.nativeId;
    }
}
