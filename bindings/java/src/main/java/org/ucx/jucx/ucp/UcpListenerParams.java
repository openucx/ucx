/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.ucp;

import java.net.InetSocketAddress;

import org.ucx.jucx.UcxParams;

public class UcpListenerParams extends UcxParams {

    @Override
    public UcpListenerParams clear() {
        super.clear();
        sockAddr = null;
        return this;
    }

    private InetSocketAddress sockAddr;

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
}
