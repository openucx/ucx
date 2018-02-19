package org.ucx.jucx.perftest;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.ucx.jucx.perftest.PerftestDataStructures.PerfParams;

public abstract class BandwidthTest extends PerftestBase {
    protected ByteBuffer buffer;

    @Override
    protected void run(PerfParams params) throws IOException {
        ctx = new PerfContext(params);
        initBuffer();
        ctx.cb = new PerftestCallback(buffer);
        super.run(params);
    }

    protected void initBuffer() {
        buffer = ByteBuffer.allocateDirect(ctx.params.size);
    }
}
