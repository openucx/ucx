package org.ucx.jucx.perftest;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.ucx.jucx.perftest.PerftestDataStructures.PerfParams;
import org.ucx.jucx.perftest.PerftestDataStructures.UcpObjects;
import org.ucx.jucx.util.Utils;

public abstract class LatencyTest extends PerftestBase {
    protected ByteBuffer sendBuff;
    protected ByteBuffer recvBuff;

    @Override
    protected abstract void execute(int iters) throws IOException;

    @Override
    protected void run(PerfParams params) throws IOException {
        ctx = new PerfContext(params);
        initBuffers();
        ctx.cb = new PerftestCallback(recvBuff);
        super.run(params);
    }

    protected void safeSendRecv(UcpObjects ucpObj,
                                long sendId,
                                long recvId) throws IOException {
        safeSend(ucpObj, sendBuff, sendId);
        safeRecv(ucpObj, recvBuff, recvId);
    }

    private void initBuffers() {
        int s = ctx.params.size;
        sendBuff = ByteBuffer.allocateDirect(s);
        recvBuff = ByteBuffer.allocateDirect(s);

        byte[] msg = Utils.generateRandomBytes(s);
        sendBuff.put(msg);
        sendBuff.flip();
    }
}
