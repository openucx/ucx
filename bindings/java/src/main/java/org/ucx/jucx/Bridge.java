/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import org.ucx.jucx.ucp.UcpContext;
import org.ucx.jucx.ucp.UcpParams;
import org.ucx.jucx.ucp.UcpWorker;
import org.ucx.jucx.ucp.UcpWorkerParams;

public class Bridge {
  private static final String UCM = "libucm.so";
  private static final String UCS = "libucs.so";
  private static final String UCT = "libuct.so";
  private static final String UCP = "libucp.so";
  private static final String JUCX = "libjucx.so";

  static {
    LoadLibrary.loadLibrary(UCM);   // UCM library
    LoadLibrary.loadLibrary(UCS);   // UCS library
    LoadLibrary.loadLibrary(UCT);   // UCT library
    LoadLibrary.loadLibrary(UCP);   // UCP library
    LoadLibrary.loadLibrary(JUCX);  // JUCP native library
  }

  // UcpContext.
  private static native long createContextNative(UcpParams params);

  private static native void cleanupContextNative(long contextId);

  // UcpWorker.
  private static native long createWorkerNative(UcpWorkerParams params, long contextId);

  private static native void releaseWorkerNative(long workerId);

  public static UcpContext createContext(UcpParams params) {
    return new UcpContext(createContextNative(params));
  }

  public static void cleanupContext(UcpContext context) {
    cleanupContextNative(context.getNativeId());
  }

  public static UcpWorker createWorker(UcpWorkerParams params, UcpContext context) {
    return new UcpWorker(createWorkerNative(params, context.getNativeId()));
  }

  public static void releaseWorker(UcpWorker worker) {
    releaseWorkerNative(worker.getNativeId());
  }
}
