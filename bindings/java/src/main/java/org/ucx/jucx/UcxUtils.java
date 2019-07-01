/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import java.nio.ByteBuffer;

public class UcxUtils {

    /**
     * Returns native address of direct byte buffer with respect of it's current position.
     */
    public static long getAddress(ByteBuffer buffer) {
        return ((sun.nio.ch.DirectBuffer) buffer).address() + buffer.position();
    }
}
