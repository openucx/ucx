/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxException;
import org.openucx.jucx.UcxNativeStruct;

import java.io.Closeable;
import java.net.InetSocketAddress;

/**
 * The listener handle is an opaque object that is used for listening on a
 * specific address and accepting connections from clients.
 */
public class UcpListener extends UcxNativeStruct implements Closeable {

    private InetSocketAddress address;
    private UcpListenerConnectionHandler connectionHandler;

    public UcpListener(UcpWorker worker, UcpListenerParams params) {
        if (params.getSockAddr() == null) {
            throw new UcxException("UcpListenerParams.sockAddr must be non-null.");
        }
        if (params.connectionHandler == null) {
            throw new UcxException("Connection handler must be set");
        }
        this.connectionHandler = params.connectionHandler;
        setNativeId(createUcpListener(params, worker.getNativeId()));
    }

    /**
     * Returns a socket address of this listener.
     */
    public InetSocketAddress getAddress() {
        if (address == null) {
            address = queryAddressNative(getNativeId());
        }
        return address;
    }

    @Override
    public void close() {
        destroyUcpListenerNative(getNativeId());
        setNativeId(null);
    }

    private native long createUcpListener(UcpListenerParams params, long workerId);

    private static native InetSocketAddress queryAddressNative(long listenerId);

    private static native void destroyUcpListenerNative(long listenerId);
}
