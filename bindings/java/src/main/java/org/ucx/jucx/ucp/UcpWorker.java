/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import java.io.Closeable;

import org.ucx.jucx.UcxNativeStruct;

/**
 * UCP worker is an opaque object representing the communication context.  The
 * worker represents an instance of a local communication resource and progress
 * engine associated with it. Progress engine is a construct that is
 * responsible for asynchronous and independent progress of communication
 * directives. The progress engine could be implement in hardware or software.
 * The worker object abstract an instance of network resources such as a host
 * channel adapter port, network interface, or multiple resources such as
 * multiple network interfaces or communication ports. It could also represent
 * virtual communication resources that are defined across multiple devices.
 * Although the worker can represent multiple network resources, it is
 * associated with a single {@link UcpContext} "UCX application context".
 * All communication functions require a context to perform the operation on
 * the dedicated hardware resource(s) and an "endpoint" to address the
 * destination.
 *
 * <p>Worker are parallel "threading points" that an upper layer may use to
 * optimize concurrent communications.
 */
public class UcpWorker extends UcxNativeStruct implements Closeable {

    public UcpWorker(UcpContext context, UcpWorkerParams params) {
        setNativeId(createWorkerNative(params, context.getNativeId()));
    }

    @Override
    public void close() {
        releaseWorkerNative(getNativeId());
        setNativeId(null);
    }

    private static native long createWorkerNative(UcpWorkerParams params, long ucpContextId);

    private static native void releaseWorkerNative(long workerId);
}
