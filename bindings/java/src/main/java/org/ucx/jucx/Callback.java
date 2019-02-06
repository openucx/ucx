/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import java.nio.ByteBuffer;

/**
 * The Callback interface must be implemented in-order to create a completion notification
 * from native code.
 */
public interface Callback {
  void onComplete(ByteBuffer data);
}
