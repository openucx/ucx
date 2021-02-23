/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxParams;

public class UcpMemMapParams extends UcxParams {
    private long flags;
    private long prot;
    private int memType;
    private long address;
    private long length;

    @Override
    public UcpMemMapParams clear() {
        super.clear();
        address = 0;
        length = 0;
        flags = 0;
        prot = 0;
        memType = 0;
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

    /**
     * Memory protection mode, e.g. {@link UcpConstants#UCP_MEM_MAP_PROT_LOCAL_READ}
     * This value is optional. If it's not set, the {@link UcpContext#memoryMap(UcpMemMapParams)}
     * routine will consider the flags as set to
     * UCP_MEM_MAP_PROT_LOCAL_READ|UCP_MEM_MAP_PROT_LOCAL_WRITE|
     * UCP_MEM_MAP_PROT_REMOTE_READ|UCP_MEM_MAP_PROT_REMOTE_WRITE.
     */
    public UcpMemMapParams setProtection(long protection) {
        this.fieldMask |= UcpConstants.UCP_MEM_MAP_PARAM_FIELD_PROT;
        this.prot = protection;
        return this;
    }

    /**
     * Memory type (for possible memory types see
     * {@link org.openucx.jucx.ucs.UcsConstants.MEMORY_TYPE})
     * It is an optimization hint to avoid memory type detection for map buffer.
     * The meaning of this field depends on the operation type.
     *
     * - Memory allocation: ({@link UcpMemMapParams#allocate()} is set) This field
     *   specifies the type of memory to allocate. If it's not set
     *   {@link org.openucx.jucx.ucs.UcsConstants.MEMORY_TYPE#UCS_MEMORY_TYPE_HOST}
     *   will be assumed by default.
     *
     * - Memory registration: This field specifies the type of memory which is
     *   pointed by {@link UcpMemMapParams#setAddress(long)}. If it's not set,
     *   or set to {@link org.openucx.jucx.ucs.UcsConstants.MEMORY_TYPE#UCS_MEMORY_TYPE_UNKNOWN},
     *   the memory type will be detected internally.
     */
    public UcpMemMapParams setMemoryType(int memoryType) {
        this.fieldMask |= UcpConstants.UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        this.memType = memoryType;
        return this;
    }
}
