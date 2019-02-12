/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import org.ucx.jucx.ucp.UcpContext;
import org.ucx.jucx.ucp.UcpParams;

public class Bridge {
    private static final String UCM  = "libucm.so";
    private static final String UCS  = "libucs.so";
    private static final String UCT  = "libuct.so";
    private static final String UCP  = "libucp.so";
    private static final String JUCX = "libjucx.so";

    static {
        LoadLibrary.loadLibrary(UCM);   // UCM library
        LoadLibrary.loadLibrary(UCS);   // UCS library
        LoadLibrary.loadLibrary(UCT);   // UCT library
        LoadLibrary.loadLibrary(UCP);   // UCP library
        LoadLibrary.loadLibrary(JUCX);  // JUCX native library
    }

    // UcpContext.
    private static native long createContextNative(UcpParams params);

    private static native void cleanupContextNative(long contextId);

    public static UcpContext createContext(UcpParams params) {
        return new UcpContext(createContextNative(params));
    }

    public static void cleanupContext(UcpContext context) {
        cleanupContextNative(context.getNativeId());
    }
}
