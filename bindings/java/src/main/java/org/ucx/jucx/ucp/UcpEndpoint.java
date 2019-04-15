package org.ucx.jucx.ucp;

import org.ucx.jucx.UcxNativeStruct;

import java.io.Closeable;

public class UcpEndpoint extends UcxNativeStruct implements Closeable {

    public UcpEndpoint(UcpWorker worker, UcpEndpointParams params) {
        setNativeId(createEndpointNative(params, worker.getNativeId()));
    }

    @Override
    public void close() {
        destroyEndpointNative(getNativeId());
        setNativeId(null);
    }

    private static native long createEndpointNative(UcpEndpointParams params, long workerId);

    private static native void destroyEndpointNative(long epId);
}
