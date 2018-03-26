/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.examples.helloworld;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public class HelloServer extends HelloWorld {
    private ServerSocket serv;

    public HelloServer() throws IOException {
        serv = new ServerSocket(port);
        serv.setReuseAddress(true);
    }

    @Override
    protected void exchangeWorkerAddress() throws IOException {
        conditionalPrint("Waiting for connections...");
        Socket sock = null;
        try {
            sock = serv.accept();
        } catch (IOException ioe) {
            throw ioe;
        } finally {
            serv.close();
        }
        conditionalPrint("Connected to: " +
                         sock.getInetAddress().getHostAddress());

        try {
            sendLocalWorkerAddress(sock);

            byte[] remoteWorkerAddress = recvRemoteWorkerAddress(sock);
            ep = worker.createEndPoint(remoteWorkerAddress);
        } catch (IOException ioe) {
            throw ioe;
        } finally {
            sock.close();
        }
    }

    @Override
    protected void usage() {
        System.out.println("Usage: ./scripts/hello_world.sh server [OPTION]...");
        super.usage();
    }

    public static void main(String[] args) {
        try {
            HelloServer server = new HelloServer();
            server.run(args);
            server.conditionalPrint("[SUCCESS] Exiting...");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void exchangeData() throws IOException {
        do {
            recv();
            while (!recvComp) {
                worker.progress();
            }
            recvComp = false;
        } while (in.hasRemaining());

        send();
        while (!sendComp) {
            worker.progress();
        }
    }
}