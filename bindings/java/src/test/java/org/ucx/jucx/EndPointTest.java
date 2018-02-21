/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;

import org.junit.Before;
import org.junit.Test;
import org.ucx.jucx.Worker.Callback;

public class EndPointTest {
    private Callback cb;

    @Before
    public void initCallback() throws Exception {
        cb = new Callback() {};
    }

    @Test
    public void testEndPointInitializationAndGetters() {
        int maxEvents = 128;
        Worker peer = null, local = null;
        EndPoint ep = null;
        try {
            peer    = new Worker(cb, maxEvents);
            local   = new Worker(cb, maxEvents);
            ep      = local.createEndPoint(peer.getWorkerAddress());
        } catch (IOException e) {
            fail(e.getMessage());
        }

        assertTrue("Wrong Worker associated with EndPoint",
                   ep.getWorker() == local);

        peer.close();
        local.destroyEndPoint(ep);
        local.close();
    }
}
