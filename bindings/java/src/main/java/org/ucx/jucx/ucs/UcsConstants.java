/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucs;

import org.ucx.jucx.NativeLibs;

public class UcsConstants {
    static {
        NativeLibs.load();
        loadConstants();
    }

    /**
     * Only the master thread can access (i.e. the thread that initialized the context;
     * multiple threads may exist and never access)
     */
    public static int UCS_THREAD_MODE_SINGLE;
    /**
     * Multiple threads can access concurrently
     */
    public static int UCS_THREAD_MODE_MULTI;

    private static native void loadConstants();
}
