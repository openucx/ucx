/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxParams;

public class UcpMemMapParams extends UcxParams {
    private long flags;
    private long address;
    private long length;

    @Override
    public UcpMemMapParams clear() {
        super.clear();
        address = 0;
        length = 0;
        flags = 0;
        return this;
    }

    public UcpMemMapParams setAddress(long address) {
        this.fieldMask |= UcpConstants.UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
        this.address = address;
        return this;
    }

    public long getAddress() {
        return address;
    }

    public UcpMemMapParams setLength(long length) {
        this.fieldMask |= UcpConstants.UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        this.length = length;
        return this;
    }

    public long getLength() {
        return length;
    }

    /**
     * Identify requirement for allocation, if passed address is not a null-pointer
     * then it will be used as a hint or direct address for allocation.
     */
    public UcpMemMapParams allocate() {
        this.fieldMask |= UcpConstants.UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        flags |= UcpConstants.UCP_MEM_MAP_ALLOCATE;
        return this;
    }

    /**
     * Complete the registration faster, possibly by not populating the pages up-front,
     * and mapping them later when they are accessed by communication routines.
     */
    public UcpMemMapParams nonBlocking() {
        this.fieldMask |= UcpConstants.UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        flags |= UcpConstants.UCP_MEM_MAP_NONBLOCK;
        return this;
    }

    /**
     * Don't interpret address as a hint: place the mapping at exactly that
     * address. The address must be a multiple of the page size.
     */
    public UcpMemMapParams fixed() {
        this.fieldMask |= UcpConstants.UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        flags |= UcpConstants.UCP_MEM_MAP_FIXED;
        return this;
    }
}
