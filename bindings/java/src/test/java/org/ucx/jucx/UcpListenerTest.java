/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import org.junit.Test;
import org.ucx.jucx.ucp.*;

import java.net.InetSocketAddress;

import static org.junit.Assert.*;

public class UcpListenerTest {
    public static final int port = Integer.parseInt(
        System.getenv().getOrDefault("JUCX_TEST_PORT", "55321"));

    @Test
    public void testCreateUcpListener() {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = context.newWorker(new UcpWorkerParams());
        InetSocketAddress ipv4 = new InetSocketAddress("0.0.0.0", port);
        try {
            UcpListener ipv4Listener = worker.newListener(new UcpListenerParams().setSockAddr(ipv4));

            assertNotNull(ipv4Listener);
            ipv4Listener.close();

            InetSocketAddress ipv6 = new InetSocketAddress("::", port);
            UcpListener ipv6Listener = worker.newListener(
                new UcpListenerParams().setSockAddr(ipv6));

            assertNotNull(ipv6Listener);
            ipv6Listener.close();
        } catch (UcxException ex) {

        } finally {
            worker.close();
            context.close();
        }

    }
}
