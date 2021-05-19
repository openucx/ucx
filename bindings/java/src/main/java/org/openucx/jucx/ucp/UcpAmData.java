/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxCallback;
import org.openucx.jucx.UcxException;
import org.openucx.jucx.ucs.UcsConstants;

import java.io.Closeable;
import java.io.IOException;

/**
 * Wrapper over received active message data. Could be one of:
 * - Internal ucx data descriptor. Need to call {@link UcpAmData#receive} to receive actual data.
 * - Actual data. Need to call {@link UcpAmData#close()} when not needed.
 */
public class UcpAmData implements Closeable {
    private final UcpWorker worker;
    private final long address;
    private final long length;
    private final long flags;

    private UcpAmData(UcpWorker worker, long address, long length, long flags) {
        this.worker = worker;
        this.address = address;
        this.length = length;
        this.flags = flags;
    }

    @Override
    public String toString() {
        return "UcpAmData{" +
            "address=" + Long.toHexString(address) +
            ", length=" + length +
            ", received=" + isDataValid() +
            '}';
    }

    /**
     * Whether actual data is received or need to call {@link UcpAmData#receive(long, UcxCallback)}
     */
    public boolean isDataValid() {
        return (flags & UcpConstants.UCP_AM_RECV_ATTR_FLAG_RNDV) == 0;
    }

    /**
     * Whether this amData descriptor can be persisted outside {@link UcpAmRecvCallback#onReceive}
     * callback by returning {@link UcsConstants.STATUS#UCS_INPROGRESS}
     */
    public boolean canPersist() {
        return (flags & UcpConstants.UCP_AM_RECV_ATTR_FLAG_DATA) != 0;
    }

    /**
     * Receive operation flags.
     */
    public long getFlags() {
        return flags;
    }

    /**
     * Get an address of received data
     */
    public long getDataAddress() {
        if (!isDataValid()) {
            throw new UcxException("Data is not received yet.");
        }
        return address;
    }

    public long getLength() {
        return length;
    }

    /**
     * Get UCX data handle descriptor to pass to {@link UcpWorker#recvAmDataNonBlocking}
     */
    public long getDataHandle() {
        return address;
    }

    public UcpRequest receive(long resultAddress, UcxCallback callback) {
        return worker.recvAmDataNonBlocking(getDataHandle(), resultAddress,
            length, callback, UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_UNKNOWN);
    }

    @Override
    public void close() throws IOException {
        if (isDataValid()) {
            worker.amDataRelease(address);
        }
    }
}
