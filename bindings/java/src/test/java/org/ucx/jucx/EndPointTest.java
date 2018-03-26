package org.ucx.jucx;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;

import org.junit.Before;
import org.junit.Test;
import org.ucx.jucx.Worker.Callback;
import org.ucx.jucx.util.TestCallback;

public class EndPointTest {
    private Callback cb;

    @Before
    public void initCallback() throws Exception {
        cb = new TestCallback();
    }

    @Test
    public void testEndPointInitializationAndGetters() {
        int maxEvents = 128;
        Worker peer = null, local = null;
        EndPoint ep = null;
        try { // Add EndPoint to example code in next commit
            peer    = new Worker(cb, maxEvents);
            local   = new Worker(cb, maxEvents);
            ep      = local.createEndPoint(peer.getWorkerAddress());
        } catch (IOException e) {
            fail(e.getMessage());
        }

        assertTrue("Wrong Worker associated with EndPoint",
                   ep.getWorker() == local);

        peer.close();
        ep.close();
        local.close();
    }
}
