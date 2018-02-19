package org.ucx.jucx.perftest;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.ucx.jucx.EndPoint;
import org.ucx.jucx.Worker;
import org.ucx.jucx.Worker.CompletionStatus;
import org.ucx.jucx.perftest.PerftestDataStructures.PerfParams;
import org.ucx.jucx.perftest.PerftestDataStructures.TcpConnection;
import org.ucx.jucx.perftest.PerftestDataStructures.UcpObjects;
import org.ucx.jucx.util.TestCallback;

public abstract class PerftestBase {
    protected PerfContext ctx;

    protected void connect() throws IOException {
        UcpObjects ucp = ctx.ucpObj;
        TcpConnection tcp = ctx.params.tcpConn;
        byte[] localAddress = ucp.worker.getWorkerAddress();

        tcp.writeInt(localAddress.length);
        tcp.write(localAddress);
        int len = tcp.readInt();
        byte[] remote = new byte[len];
        tcp.read(remote);
        ucp.setEndPoint(remote);
    }

    protected void warmup() throws IOException {
        execute(ctx.params.warmupIter);
    }

    protected void barrier(TcpConnection tcp) throws IOException {
        tcp.barrier(this instanceof LatencyServer ||
                    this instanceof BandwidthServer);
    }

    protected void run(PerfParams params) throws IOException {
        setup();
        connect();

        TcpConnection tcp = params.tcpConn;

        barrier(tcp);
        warmup();
        barrier(tcp);

        execute(params.maxIter);
        barrier(tcp);
    }

    protected boolean done() {
        return ctx.measure.done();
    }

    protected void safeRecv(UcpObjects ucpObj,
                            ByteBuffer recvBuff,
                            long id) throws IOException {
        EndPoint    ep      = ucpObj.endPoint;
        Worker      worker  = ucpObj.worker;
        do {
            ep.streamRecv(recvBuff, id);
            do {
                worker.progress();
            } while (!ctx.cb.recvComplete);
            ctx.cb.recvComplete = false;
        } while (recvBuff.hasRemaining());
    }

    protected void safeSend(UcpObjects ucpObj,
                            ByteBuffer sendBuff,
                            long id) throws IOException {
        EndPoint    ep      = ucpObj.endPoint;
        Worker      worker  = ucpObj.worker;
        ep.streamSend(sendBuff, id);
        do {
            worker.progress();
        } while (!ctx.cb.sendComplete);
        ctx.cb.sendComplete = false;
    }

    void close() {
        ctx.ucpObj.close();
    }

    private void setup() throws IOException {
        ctx.ucpObj = new UcpObjects();
        ctx.ucpObj.setWorker(ctx.cb, ctx.params.events);
    }

    protected abstract void execute(int iters) throws IOException;


    protected static class PerftestCallback extends TestCallback {
        protected   boolean     sendComplete;
        protected   boolean     recvComplete;
        private     ByteBuffer  recvBuff;

        public PerftestCallback(ByteBuffer buff) {
            sendComplete = false;
            recvComplete = false;
            recvBuff     = buff;
        }

        @Override
        public void sendCompletionHandler(long requestId,
                                          CompletionStatus status) {
            sendComplete = true;
        }

        @Override
        public void recvCompletionHandler(long requestId,
                                          CompletionStatus status,
                                          int length) {
            recvComplete = true;
            recvBuff.position(recvBuff.position() + length);
        }
    }
}
