package org.ucx.jucx.perftest;

import java.io.IOException;

import org.ucx.jucx.perftest.PerftestDataStructures.PerfMeasurements;
import org.ucx.jucx.perftest.PerftestDataStructures.PerfParams;
import org.ucx.jucx.util.Utils;

public class BandwidthClient extends BandwidthTest implements PerftestClient {
    @Override
    protected void warmup() throws IOException {
        int iters = ctx.params.warmupIter;

        for (int i = 0; i < iters; i++) {
            safeSend(ctx.ucpObj, buffer, i);
            if (ctx.print) {
                System.out.println("Iteration #" + i + " in warmup");
            }

//            System.nanoTime();
        }
    }

    @Override
    protected void initBuffer() {
        super.initBuffer();
        buffer.put(Utils.generateRandomBytes(buffer.capacity()));
        buffer.flip();
    }

    @Override
    protected void execute(int iters) throws IOException {
        PerfParams params = ctx.params;
        PerfMeasurements measure = ctx.measure;
        measure.setPerfMeasurements(params.maxTimeSecs, params.reportInterval);
        measure.setTimesArray(iters);

        for (int i = 0; !done(); i++) {
            safeSend(ctx.ucpObj, buffer, i);
            measure.currTime = System.nanoTime();
            measure.setMeasurement(i, 1);

            if (ctx.print) {
                System.out.println("Iteration #" + (i + 1) + " in main loop");
            }
        }

        measure.endTime = measure.currTime;

        System.out.println("\nBandwidth Test Results:");
        printResults(ctx);
    }
}
