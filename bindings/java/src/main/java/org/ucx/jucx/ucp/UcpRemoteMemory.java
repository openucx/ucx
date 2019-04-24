/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import java.io.Serializable;
import java.nio.ByteBuffer;

/**
 * Class that represents remote memory with it's address and remote key buffer.
 */
public class UcpRemoteMemory implements Serializable {

    private ByteBuffer remoteKey;

    private long address;

    public ByteBuffer getRemoteKey() {
        return remoteKey;
    }

    public long getAddress() {
        return address;
    }

    public UcpRemoteMemory(ByteBuffer remoteKey, long address) {
        this.address = address;
        this.remoteKey = remoteKey;
    }

    public UcpRemoteMemory(UcpMemory memory) {
        this.address = memory.getAddress();
        this.remoteKey = memory.getRemoteKeyBuffer();
    }
}
