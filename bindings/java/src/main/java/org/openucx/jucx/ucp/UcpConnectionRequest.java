/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxNativeStruct;

import java.net.InetSocketAddress;

/**
 * A server-side handle to incoming connection request. Can be used to create an
 * endpoint which connects back to the client.
 */
public class UcpConnectionRequest extends UcxNativeStruct {

    private InetSocketAddress clientAddress;

    /**
     * The address of the remote client that sent the connection request to the server.
     */
    public InetSocketAddress getClientAddress() {
        return clientAddress;
    }

    private UcpConnectionRequest(long nativeId, InetSocketAddress clientAddress) {
        setNativeId(nativeId);
        this.clientAddress = clientAddress;
    }
}
