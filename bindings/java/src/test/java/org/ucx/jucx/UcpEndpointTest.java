/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import org.junit.Test;
import org.ucx.jucx.ucp.*;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.Enumeration;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class UcpEndpointTest {
    @Test
    public void testConnectToListenerByWorkerAddr() {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = new UcpWorker(context, new UcpWorkerParams());
        InetSocketAddress listenerSocket = new InetSocketAddress("0.0.0.0", UcpListenerTest.port);
        UcpListener ucpListener = new UcpListener(worker,
            new UcpListenerParams().setSockAddr(listenerSocket));

        UcpEndpointParams epParams = new UcpEndpointParams().setUcpAddress(worker.getAddress())
            .setPeerErrorHadnlingMode().setNoLoopbackMode();
        UcpEndpoint endpoint = new UcpEndpoint(worker, epParams);
        assertNotNull(endpoint.getNativeId());

        endpoint.close();
        ucpListener.close();
        worker.close();
        context.close();
    }

    @Test
    public void testConnectToListenerBySocketAddr() throws SocketException {
        UcpContext context = new UcpContext(new UcpParams().requestStreamFeature());
        UcpWorker worker = new UcpWorker(context, new UcpWorkerParams());
        // Iterate over each network interface - got it's sockaddr - try to instantiate listener
        // And pass this sockaddr to endpoint.
        Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
        boolean success = false;
        while (!success && interfaces.hasMoreElements()) {
            NetworkInterface networkInterface = interfaces.nextElement();
            Enumeration<InetAddress> inetAddresses = networkInterface.getInetAddresses();
            while (inetAddresses.hasMoreElements()) {
                InetAddress inetAddress = inetAddresses.nextElement();
                if (!inetAddress.isLoopbackAddress()) {
                    try {
                        InetSocketAddress addr = new InetSocketAddress(inetAddress,
                            UcpListenerTest.port);
                        UcpListener ucpListener = new UcpListener(worker,
                            new UcpListenerParams().setSockAddr(addr));
                        UcpEndpointParams epParams =
                            new UcpEndpointParams().setSocketAddress(addr);
                        UcpEndpoint endpoint = new UcpEndpoint(worker, epParams);
                        assertNotNull(endpoint.getNativeId());
                        success = true;
                        endpoint.close();
                        ucpListener.close();
                        break;
                    } catch (UcxException ex) {

                    }
                }
            }
        }
        assertTrue(success);

        worker.close();
        context.close();
    }
}
