/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import java.nio.ByteBuffer;

public class UcxUtils {

    /**
     * Returns native address of the current position of a direct byte buffer.
     */
    public static long getAddress(ByteBuffer buffer) {
        return ((sun.nio.ch.DirectBuffer) buffer).address() + buffer.position();
    }
}
