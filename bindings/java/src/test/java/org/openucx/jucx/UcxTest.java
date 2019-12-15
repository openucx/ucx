/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import java.io.Closeable;
import java.io.IOException;
import java.util.Stack;

abstract class UcxTest {
    // Stack of closable resources (context, worker, etc.) to be closed at the end.
    protected static Stack<Closeable> resources = new Stack<>();

    protected void closeResources() {
        while (!resources.empty()) {
            try {
                resources.pop().close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
