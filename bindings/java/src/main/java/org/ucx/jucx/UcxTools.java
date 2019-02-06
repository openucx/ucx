/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

public class UcxTools {
  //CHECKSTYLE:OFF: checkstyle:all
  public static long UCS_BIT(long i){ return (1L << i); }
  public static int UCS_BIT(int i){ return (1 << i); }
  //CHECKSTYLE:ON: checkstyle:all

  public enum UcsTreadMode {
    // Only the master thread can access (i.e. the thread that initialized the context;
    // multiple threads may exist and never access)
    UCS_THREAD_MODE_SINGLE,
    UCS_THREAD_MODE_SERIALIZED, // Multiple threads can access, but only one at a time
    UCS_THREAD_MODE_MULTI,      // Multiple threads can access concurrently
    UCS_THREAD_MODE_LAST
  }

}
