/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxNativeStruct;
import org.openucx.jucx.ucp.UcpListener;
import org.openucx.jucx.ucs.UcsConstants;

import java.net.InetSocketAddress;

/**
 * A server-side handle to incoming connection request. Can be used to create an
 * endpoint which connects back to the client.
 */
public class UcpConnectionRequest extends UcxNativeStruct {

    private InetSocketAddress clientAddress;
    private UcpListener listener;
    private long clientId;

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

    /**
     * Reject the incoming connection request and release associated resources. If
     * the remote initiator endpoint has set an {@link UcpEndpointParams#setErrorHandler},
     * it will be invoked with status {@link UcsConstants.STATUS#UCS_ERR_REJECTED}.
     */
    public void reject() {
        rejectConnRequestNative(listener.getNativeId(), getNativeId());
    }

    /**
     * Client id of remote endpoint.
     */
    public long getClientId() {
        return clientId;
    }

    private static native void rejectConnRequestNative(long listenerId, long connRequestId);
}
