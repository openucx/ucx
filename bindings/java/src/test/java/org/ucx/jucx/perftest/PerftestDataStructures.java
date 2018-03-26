package org.ucx.jucx.perftest;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.net.Socket;

import org.ucx.jucx.EndPoint;
import org.ucx.jucx.Worker;
import org.ucx.jucx.Worker.Callback;
import org.ucx.jucx.util.Time;

public class PerftestDataStructures {

    public static enum PerfTestType {
        JUCX_TEST_PINGPONG, // for latency test
        JUCX_TEST_BANDWIDTH // for BW test
    }

    public static enum PerfDataType {
        JUCX_DATATYPE_CONTIG,
        JUCX_DATATYPE_IOV
    }

    public static class PerfParams implements Serializable {
        private static final long serialVersionUID = 1L;

        PerfTestType  testType;       // Test communication type
        int           maxOutstanding; // Maximal number of outstanding sends
        int           warmupIter;     // Number of warm-up iterations
        int           maxIter;        // Iterations limit, 0 - unlimited
        double        maxTimeSecs;    // Time limit (seconds), 0 - unlimited
        int           size;           // Message size
        double        reportInterval; // Interval at which to call the report
                                      // callback
        PerfDataType  sendType;
        PerfDataType  recvType;
        UcpObjects    ucp;
        String        filename;
        TcpConnection tcpConn;
        int           events;
        boolean       print;

        PerfParams() {
            testType        = PerfTestType.JUCX_TEST_PINGPONG;
            maxOutstanding  = 1;
            warmupIter      = 10000;
            maxIter         = 1000000;
            size            = 64;
            events          = 200;
            sendType        = PerfDataType.JUCX_DATATYPE_CONTIG;
            recvType        = PerfDataType.JUCX_DATATYPE_CONTIG;
            maxTimeSecs     = 0.0;
            reportInterval  = 1.0;
            print           = false;
            ucp             = null;
            filename        = null;
            tcpConn         = null;
        }
    }

    public static class PerfMeasurements {
        // Time
        long startTime;
        long prevTime;
        long currTime;
        long endTime;
        long reportInterval;

        // Space
        long bytes;
        int  iters;
        int  msgs;
        int  maxIter;

        private long[] times;

        PerfMeasurements(int maxIter) {
            this.maxIter = maxIter;
        }

        void setPerfMeasurements(double secs, double report) {
            currTime = prevTime = startTime = System.nanoTime();
            endTime = (secs == 0.0) ?
                      Long.MAX_VALUE : (Time.secsToNanos(secs) + startTime);
            reportInterval = Time.secsToNanos(report);
            bytes = iters = msgs = 0;
        }

        void setTimesArray(int size) {
            times = new long[size];
        }

        void setMeasurement(int index, int iters) {
            this.iters += iters;
            setTimeSample(index);
            prevTime = currTime;
        }

        void setCurrentMeasurement(int index, int iters, int msgs, long bytes) {
            setSpaceSample(iters, msgs, bytes);
            setTimeSample(index);
        }

        boolean done() {
            return (currTime >= endTime) || (iters >= maxIter);
        }

        long[] timeSamples() {
            return times.clone();
        }

        private void setSpaceSample(int iters, int msgs, long bytes) {
            this.iters += iters;
            this.msgs += msgs;
            this.bytes += bytes;
        }

        private void setTimeSample(int index) {
            times[index] = currTime - prevTime;
        }
    }

    public static class UcpObjects {
        Worker   worker   = null;
        EndPoint endPoint = null;

        public void setWorker(Callback cb, int events) throws IOException {
            worker = new Worker(cb, events);
        }

        public void setEndPoint(byte[] remoteAddress) throws IOException {
            endPoint = worker.createEndPoint(remoteAddress);
        }

        public void close() {
            worker.close();
        }
    }

    public static class TcpConnection {
        private Socket              sock;
        private ObjectInputStream   inStream;
        private ObjectOutputStream  outStream;

        TcpConnection(Socket socket) throws IOException {
            sock        = socket;
            outStream   = new ObjectOutputStream(sock.getOutputStream());
            inStream    = new ObjectInputStream(sock.getInputStream());
        }

        void close() {
            try {
                inStream.close();
                outStream.close();
                sock.close();
            } catch (IOException e) {
                // Close quietly
            }
        }

        void writeInt(int n) throws IOException {
            outStream.writeInt(n);
            outStream.flush();
        }

        void write(byte[] data) throws IOException {
            outStream.write(data);
            outStream.flush();
        }

        void writeObject(Object obj) throws IOException {
            outStream.writeObject(obj);
            outStream.flush();
        }

        int readInt() throws IOException {
            return inStream.readInt();
        }

        void read(byte[] data) throws IOException {
            inStream.readFully(data);
        }

        Object readObject() throws IOException, ClassNotFoundException {
            return inStream.readObject();
        }

        void barrier(boolean server) throws IOException {
            if (server) {
                serverBarrier();
            }
            else {
                clientBarrier();
            }
        }

        private void clientBarrier() throws IOException {
            int x = 0;
            outStream.write(x);
            outStream.flush();
            x = inStream.read();
        }

        private void serverBarrier() throws IOException {
            int x;
            x = inStream.read();
            outStream.write(x);
            outStream.flush();
        }
    }
}
