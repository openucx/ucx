/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import java.io.Closeable;

import org.ucx.jucx.Bridge;
import org.ucx.jucx.UcxNativeStruct;

/**
 * UCP application context (or just a context) is an opaque handle that holds a
 * UCP communication instance's global information.  It represents a single UCP
 * communication instance.  The communication instance could be an OS process
 * (an application) that uses UCP library.  This global information includes
 * communication resources, endpoints, memory, temporary file storage, and
 * other communication information directly associated with a specific UCP
 * instance.  The context also acts as an isolation mechanism, allowing
 * resources associated with the context to manage multiple concurrent
 * communication instances. For example, users using both MPI and OpenSHMEM
 * sessions simultaneously can isolate their communication by allocating and
 * using separate contexts for each of them. Alternatively, users can share the
 * communication resources (memory, network resource context, etc.) between
 * them by using the same application context. A message sent or a RMA
 * operation performed in one application context cannot be received in any
 * other application context.
 */
public class UcpContext extends UcxNativeStruct implements Closeable {
    public UcpContext(long nativeId) {
        super(nativeId);
    }

    @Override
    public void close() {
        Bridge.cleanupContext(this);
        this.setNativeId(0L);
    }
}
