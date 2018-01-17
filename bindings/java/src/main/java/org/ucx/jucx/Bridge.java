/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

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
        LoadLibrary.loadLibrary(JUCX);  // JUCP native library
    }
}
