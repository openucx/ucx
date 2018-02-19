/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.examples.helloworld;

import java.io.IOException;
import java.net.Socket;

public class HelloClient extends HelloWorld {
    private String host = "127.0.0.1";

    @Override
    protected void exchangeWorkerAddress() throws IOException {
        // TCP connect to peer in order to exchange worker address
        Socket sock = new Socket(host, port);

        // Print progress message only if not in quiet mode
        conditionalPrint("Connected to: " + sock.getInetAddress().getHostAddress());

        // Received remote address through TCP socket
        @SuppressWarnings("unused")
        byte[] remoteWorkerAddress = recvRemoteWorkerAddress(sock);

        // Send local Worker address through TCP socket
        sendLocalWorkerAddress(sock);

        sock.close();
    }

    @Override
    protected void usage() {
        System.out.println("Usage: ./scripts/hello_world.sh client "
                           + "[<Host_IP_address>] [OPTION]...");
        System.out.println("Default Host_IP_address: 127.0.0.1");
        super.usage();
    }

    @Override
    protected void parseArgs(String[] args) {
        super.parseArgs(args);
        String h = options.getNonOptionArgument();
        if (h != null) {
            host = h;
        }
    }

    public static void main(String[] args) {
        HelloClient client = new HelloClient();
        try {
            client.run(args);
            client.conditionalPrint("[SUCCESS] Exiting...");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
