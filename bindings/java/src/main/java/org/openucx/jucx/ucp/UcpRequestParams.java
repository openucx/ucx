/*
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import static org.openucx.jucx.ucs.UcsConstants.MEMORY_TYPE.UCS_MEMORY_TYPE_UNKNOWN;

/**
 * Additional request parameters
 */
public class UcpRequestParams {

    private int  memType;
    private long memHandle;


    public UcpRequestParams() {
        this.memType = UCS_MEMORY_TYPE_UNKNOWN;
    }

    /**
     * Memory type of the buffer.
     * An optimization hint to avoid memory type detection for request buffer.
     */
    public UcpRequestParams setMemoryType(int memType) {
        this.memType = memType;
        return this;
    }

    /**
     * Memory handle for pre-registered buffer.
     * If the handle is provided, protocols that require registered memory can
     * skip the registration step. As a result, the communication request
     * overhead can be reduced and the request can be completed faster.
     */
    public UcpRequestParams setMemoryHandle(UcpMemory memHandle) {
        this.memHandle = memHandle.getNativeId();
        return this;
    }

}
