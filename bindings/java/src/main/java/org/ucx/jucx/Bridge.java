/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import org.ucx.jucx.Worker.CompletionQueue;

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

    private static native long createWorkerNative(int maxCompletions,
            CompletionQueue compQueue, Worker worker);

    static long createWorker(final int maxCompletions,
            final CompletionQueue compQueue, final Worker worker) {
        return createWorkerNative(maxCompletions, compQueue, worker);
    }

    private static native void releaseWorkerNative(long workerNativeId);

    static void releaseWorker(final Worker worker) {
        releaseWorkerNative(worker.getNativeId());
    }
}
