/*
 * Copyright (C) Mellanox Technologies Ltd. 2020. ALL RIGHTS RESERVED.
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
 * - Internal ucx data descriptor. Need to call {@link UcpAmData#receive }
 * to receive actual data.
 * - Actual data. Need to call {@link UcpAmData#close()} when not needed.
 */
public class UcpAmData implements Closeable {
    private final UcpAmRecvCallback ucpAmRecvCallback;
    private final long address;
    private final long length;
    private final long flags;

    private UcpAmData(UcpAmRecvCallback ucpAmRecvCallback,
                      long address, long length, long flags) {
        this.ucpAmRecvCallback = ucpAmRecvCallback;
        this.address = address;
        this.length = length;
        this.flags = flags;
    }

    @Override
    public String toString() {
        return "UcpAmData{" +
            "address=" + address +
            ", length=" + length +
            ", received=" + isReceived() +
            '}';
    }

    public boolean isReceived() {
        return (flags & UcpConstants.UCP_AM_RECV_ATTR_FLAG_DATA) != 0;
    }

    public long getDataAddress() {
        if (!isReceived()) {
            throw new UcxException("Data is not received yet.");
        }
        return address;
    }

    public long getLength() {
        return length;
    }

    public long getDataHandle() {
        if (isReceived()) {
            throw new UcxException("Handle is invalid, since data is received.");
        }
        return address;
    }

    public UcpRequest receive(long resultAddress, UcxCallback callback) {
        return ucpAmRecvCallback.worker.recvAmDataNonBlocking(address, resultAddress, length,
            callback, UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_UNKNOWN);
    }

    @Override
    public void close() throws IOException {
        if (isReceived()) {
            ucpAmRecvCallback.worker.amDataRelease(address);
        }
    }
}
