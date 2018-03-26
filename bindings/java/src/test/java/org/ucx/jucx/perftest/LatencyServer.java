package org.ucx.jucx.perftest;

import java.io.IOException;

public class LatencyServer extends LatencyTest {
    @Override
    protected void execute(int iters) throws IOException {
        safeRecv(ctx.ucpObj, recvBuff, 0);
        recvBuff.flip();

        for (int i = 0; i < iters - 1; i++) {
            safeSendRecv(ctx.ucpObj, i, i + 1);
            recvBuff.flip();
        }

        safeSend(ctx.ucpObj, sendBuff, iters - 1);
    }
}
