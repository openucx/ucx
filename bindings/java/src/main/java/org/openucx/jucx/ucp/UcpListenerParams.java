/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import java.net.InetSocketAddress;

import org.openucx.jucx.UcxParams;

public class UcpListenerParams extends UcxParams {

    @Override
    public UcpListenerParams clear() {
        super.clear();
        sockAddr = null;
        connectionHandler = null;
        return this;
    }

    private InetSocketAddress sockAddr;

    UcpListenerConnectionHandler connectionHandler;

    /**
     *  An address, on which {@link UcpListener} would bind.
     */
    public UcpListenerParams setSockAddr(InetSocketAddress sockAddr) {
        this.sockAddr = sockAddr;
        this.fieldMask |= UcpConstants.UCP_LISTENER_PARAM_FIELD_SOCK_ADDR;
        return this;
    }

    public InetSocketAddress getSockAddr() {
        return sockAddr;
    }

    /**
     *  Handler of an incoming connection request in a client-server connection flow.
     */
    public UcpListenerParams setConnectionHandler(UcpListenerConnectionHandler handler) {
        this.connectionHandler = handler;
        this.fieldMask |= UcpConstants.UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
        return this;
    }
}
