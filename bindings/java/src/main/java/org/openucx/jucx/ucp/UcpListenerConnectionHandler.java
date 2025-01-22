/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

/**
 * A server-side handle to incoming connection request. Can be used to create an
 * endpoint which connects back to the client.
 */
public interface UcpListenerConnectionHandler {
    /**
     * This callback routine is invoked on the server side to handle incoming
     * connections from remote clients.
     * @param connectionRequest - native pointer to connection request, that could be used
     *                    in {@link UcpEndpointParams#setConnectionRequest(
     *                                                     UcpConnectionRequest connectionRequest)}
     */
    void onConnectionRequest(UcpConnectionRequest connectionRequest);
}
