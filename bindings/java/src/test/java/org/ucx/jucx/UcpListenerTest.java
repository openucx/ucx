/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import org.junit.Test;
import org.ucx.jucx.ucp.*;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.*;

public class UcpListenerTest {
    private static final int port = Integer.parseInt(
        System.getenv().getOrDefault("JUCX_TEST_PORT", "55321"));

    @Test
    public void testCreateUcpListener() {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = new UcpWorker(context, new UcpWorkerParams());
        InetSocketAddress ipv4 = new InetSocketAddress("0.0.0.0", port);
        UcpListener ipv4Listener = new UcpListener(worker,
            new UcpListenerParams().setSockAddr(ipv4));

        assertNotNull(ipv4Listener);
        ipv4Listener.close();

        InetSocketAddress ipv6 = new InetSocketAddress("::", port);
        UcpListener ipv6Listener = new UcpListener(worker,
            new UcpListenerParams().setSockAddr(ipv6));

        assertNotNull(ipv6Listener);
        ipv6Listener.close();
        worker.close();
        context.close();
    }
}
