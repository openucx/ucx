/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucs;

import org.openucx.jucx.NativeLibs;

public class UcsConstants {
    static {
        load();
    }

    public static class ThreadMode {
        static {
            load();
        }
        /**
         * Multiple threads can access concurrently
         */
        public static int UCS_THREAD_MODE_MULTI;
    }

    private static void load() {
        NativeLibs.load();
        loadConstants();
    }

    private static native void loadConstants();
}
