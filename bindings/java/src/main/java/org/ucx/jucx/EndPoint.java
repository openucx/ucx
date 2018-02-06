/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import java.io.IOException;

public class EndPoint {
    private long    nativeId;
    private Worker  localWorker;
    private byte[]  remoteWorkerAddress;
    private boolean closed;

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
    public byte[] getRemoteWorkerAddress() {
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
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;

        EndPoint other = (EndPoint) obj;

        return nativeId == other.nativeId;
    }
}
