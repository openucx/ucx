package org.ucx.jucx.perftest;

import org.ucx.jucx.perftest.PerftestBase.PerftestCallback;
import org.ucx.jucx.perftest.PerftestDataStructures.PerfMeasurements;
import org.ucx.jucx.perftest.PerftestDataStructures.PerfParams;
import org.ucx.jucx.perftest.PerftestDataStructures.UcpObjects;

public class PerfContext {
    PerfParams       params;
    UcpObjects       ucpObj;
    PerftestCallback cb;
    PerfMeasurements measure;
    boolean          print;

    PerfContext(PerfParams params) {
        this.params     = params;
        this.measure    = new PerfMeasurements(params.maxIter);
        this.print      = params.print;
        this.ucpObj     = null;
        this.cb         = null;
    }
}
