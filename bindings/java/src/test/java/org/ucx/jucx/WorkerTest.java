package org.ucx.jucx;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.Random;

import org.junit.Before;
import org.junit.Test;
import org.ucx.jucx.Worker.Callback;

/**
 * Worker unit tests class
 */
public class WorkerTest {
    private Callback cb;

    @Before
    public void initCallback() {
        cb = new Callback() {};
    }

    @Test
    public void testMultipleWorkersInitialization() {
        int numOfWorkers = 10;
        int maxEvents = 128;
        Worker[] workers = new Worker[numOfWorkers];

        for (int i = 0; i < numOfWorkers; i++) {
            try {
                workers[i] = new Worker(cb, maxEvents);
            } catch (IOException e) {
                fail(e.getMessage());
            }
        }

        for (int i = 0; i < workers.length; i++) {
            workers[i].close();
        }
    }

    @Test
    public void testWorkerFieldsAndGetters() {
        int maxEvents = 128;
        Worker worker = null;
        long nativeId = -1;
        byte[] workerAddress = null;

        try {
            worker = new Worker(cb, maxEvents);
            nativeId = worker.getNativeId();
            workerAddress = worker.getWorkerAddress();
        } catch (IOException e) {
            fail(e.getMessage());
        }

        assertTrue("Worker fields initialization failed",
                nativeId > 0 &&
                workerAddress != null &&
                workerAddress.length > 0);

        if (nativeId <= 0 || workerAddress == null ||
                workerAddress.length == 0) {
            fail("Worker fields initialization failed");
        }

        worker.close();
    }

    @SuppressWarnings({ "resource", "unused" })
    @Test(expected = IllegalArgumentException.class)
    public void testWorkerCbNullInitialization() {
        // Random legal event queue size
        int maxEvents = new Random().nextInt(Worker.MAX_QUEUED_EVENTS) + 1;
        try {
            Worker worker = new Worker(null, maxEvents);
        } catch (IOException e) {
            fail(e.getMessage());
        }
    }

    @SuppressWarnings({ "resource", "unused" })
    @Test(expected = IllegalArgumentException.class)
    public void testWorkerQueueSizeNonPositiveInitialization() {
        try {
            Worker worker = new Worker(cb, 0);
        } catch (IOException e) {
            fail(e.getMessage());
        }
    }

    @SuppressWarnings({ "resource", "unused" })
    @Test(expected = IllegalArgumentException.class)
    public void testWorkerQueueSizeExceedingInitialization() {
        try {
            Worker worker = new Worker(cb, Worker.MAX_QUEUED_EVENTS + 1);
        } catch (IOException e) {
            fail(e.getMessage());
        }
    }
}
