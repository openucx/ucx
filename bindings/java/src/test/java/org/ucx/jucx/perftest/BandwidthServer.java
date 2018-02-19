package org.ucx.jucx.perftest;

import java.io.IOException;

public class BandwidthServer extends BandwidthTest {
    @Override
    protected void execute(int iters) throws IOException {
        for (int i = 0; i < iters; i++) {
            safeRecv(ctx.ucpObj, buffer, i);
            buffer.flip();
        }
    }
}
