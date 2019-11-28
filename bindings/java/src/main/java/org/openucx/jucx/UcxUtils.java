/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;

public class UcxUtils {

    private static final Constructor<?> directBufferConstructor;

    static {
        try {
            Class<?> classDirectByteBuffer = Class.forName("java.nio.DirectByteBuffer");
            directBufferConstructor = classDirectByteBuffer.getDeclaredConstructor(long.class,
                int.class);
            directBufferConstructor.setAccessible(true);
        } catch (Exception e) {
            throw new UcxException(e.getMessage());
        }
    }

    /**
     * Returns view of underlying memory region as a ByteBuffer.
     * @param address - address of start of memory region
     */
    public static ByteBuffer getByteBufferView(long address, int length)
        throws IllegalAccessException, InvocationTargetException, InstantiationException {
        return (ByteBuffer)directBufferConstructor.newInstance(address, length);
    }

    /**
     * Returns native address of the current position of a direct byte buffer.
     */
    public static long getAddress(ByteBuffer buffer) {
        return ((sun.nio.ch.DirectBuffer) buffer).address() + buffer.position();
    }
}
