/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.UcxNativeStruct;

import java.io.Closeable;
import java.nio.ByteBuffer;

/**
 * Memory handle is an opaque object representing a memory region allocated
 * through UCP library, which is optimized for remote memory access
 * operations (zero-copy operations). The memory could be registered
 * to one or multiple network resources that are supported by UCP,
 * such as InfiniBand, Gemini, and others.
 */
public class UcpMemory extends UcxNativeStruct implements Closeable {

    private UcpContext context;

    private ByteBuffer data;

    private long address;

    private long length;

    private int memType;

    /**
     * To prevent construct outside of JNI.
     */
    private UcpMemory(long nativeId, UcpContext context, long address, long length, int memType) {
        setNativeId(nativeId);
        this.address = address;
        this.length = length;
        this.memType = memType;
        this.context = context;
    }

    /**
     * This routine unmaps a user specified memory segment.
     * When the function returns, the {@code data} and associated
     * "remote key" will be invalid and cannot be used with any UCP routine.
     * Another well know terminology for the "unmap" operation that is typically
     * used in the context of networking is memory "de-registration". The UCP
     * library de-registers the memory the available hardware so it can be returned
     * back to the operation system.
     */
    public void deregister() {
        unmapMemoryNative(context.getNativeId(), getNativeId());
        setNativeId(null);
        data = null;
    }

    /**
     * This routine allocates memory buffer and packs into the buffer
     * a remote access key (RKEY) object. RKEY is an opaque object that provides
     * the information that is necessary for remote memory access.
     * This routine packs the RKEY object in a portable format such that the
     * object can be "unpacked" on any platform supported by the
     * UCP library.
     * RKEYs for InfiniBand and Cray Aries networks typically includes
     * InfiniBand and Aries key.
     * In order to enable remote direct memory access to the memory associated
     * with the memory handle the application is responsible for sharing the RKEY with
     * the peers that will initiate the access.
     */
    public ByteBuffer getRemoteKeyBuffer() {
        ByteBuffer rKeyBuffer = getRkeyBufferNative(context.getNativeId(), getNativeId());
        // 1. Allocating java native ByteBuffer (managed by java's reference count cleaner).
        ByteBuffer result = ByteBuffer.allocateDirect(rKeyBuffer.capacity());
        // 2. Copy content of native ucp address to java's buffer.
        result.put(rKeyBuffer);
        result.clear();
        // 3. Release an address of the worker object. Memory allocated in JNI must be freed by JNI.
        releaseRkeyBufferNative(rKeyBuffer);
        return result;
    }

    /**
     * To keep reference to user's ByteBuffer so it won't be cleaned by refCount cleaner.
     * @param data
     */
    void setByteBufferReference(ByteBuffer data) {
        this.data = data;
    }

    /**
     * Address of registered memory.
     */
    public long getAddress() {
        return address;
    }

    /**
     * Length of registered memory
     */
    public long getLength() {
        return length;
    }

    /**
     * Type of allocated memory.
     */
    public int getMemType() {
        return memType;
    }

    private static native void unmapMemoryNative(long contextId, long memoryId);

    private static native ByteBuffer getRkeyBufferNative(long contextId, long memoryId);

    private static native void releaseRkeyBufferNative(ByteBuffer rkey);

    @Override
    public void close() {
        deregister();
    }
}
