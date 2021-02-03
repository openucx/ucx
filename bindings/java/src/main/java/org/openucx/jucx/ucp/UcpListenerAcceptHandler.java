/*
 * Copyright (C) Mellanox Technologies Ltd. 2021. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

/**
 * A callback for accepting client/server connections on a listener
 */
public interface UcpListenerAcceptHandler {
    /**
     * This callback routine is invoked on the server side upon creating a connection
     * to a remote client.
     * The user is responsible for releasing the endpoint handle using the
     * {@link UcpEndpoint#close()}
     * @param ep - Handle to a newly created endpoint which is connected
     *             to the remote peer which has initiated the connection.
     */
    void onAccept(UcpEndpoint ep);
}
