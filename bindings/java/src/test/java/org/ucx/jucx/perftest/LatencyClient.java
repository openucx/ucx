package org.ucx.jucx.perftest;

import java.io.IOException;

import org.ucx.jucx.perftest.PerftestDataStructures.PerfMeasurements;
import org.ucx.jucx.perftest.PerftestDataStructures.PerfParams;
import org.ucx.jucx.util.Utils;

public class LatencyClient extends LatencyTest implements PerftestClient {
    @Override
    protected void warmup() throws IOException {
        PerfParams params = ctx.params;
        int iters = params.warmupIter;

        for (int i = 0; i < iters; i++) {
            safeSendRecv(ctx.ucpObj, i, i);
            recvBuff.flip();

            if (ctx.print) {
                System.out.println("Iteration #" + i + " in warmup loop");
            }

            System.nanoTime();
        }
    }

    @Override
    protected void execute(int iters) throws IOException {
        PerfParams params = ctx.params;
        PerfMeasurements measure = ctx.measure;
        measure.setPerfMeasurements(params.maxTimeSecs, params.reportInterval);
        measure.setTimesArray(iters);

        for (int i = 0; !done(); i++) {
            safeSendRecv(ctx.ucpObj, i, i);
            recvBuff.flip();

            measure.currTime = System.nanoTime();
            measure.setMeasurement(i, 1);

            if (ctx.print) {
                System.out.println("Iteration #" + i + " in main loop");
                System.out.println("Received message: "
                                   + Utils.getByteBufferAsString(recvBuff));
            }
        }

        measure.endTime = measure.currTime;

        System.out.println("\nLatency Test Results:");
        printResults(ctx);
    }
}
