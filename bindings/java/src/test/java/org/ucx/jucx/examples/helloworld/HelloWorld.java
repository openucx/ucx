/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.examples.helloworld;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.ByteBuffer;

import org.ucx.jucx.EndPoint;
import org.ucx.jucx.Worker;
import org.ucx.jucx.Worker.CompletionStatus;
import org.ucx.jucx.util.Options;
import org.ucx.jucx.util.TestCallback;

public abstract class HelloWorld {
    protected int       port    = 12345;
    protected Worker    worker  = null;
    protected Options   options = null;
    protected boolean   quiet   = false;
    protected EndPoint  ep      = null;
    protected ByteBuffer out    = null;
    protected ByteBuffer in     = null;
    protected boolean sendComp  = false;
    protected boolean recvComp  = false;

    protected final String message = "Hello from " + this.getClass().getSimpleName();

    protected void init() throws IOException {
        // Allocate Worker object with completion queue of size 128
        out = ByteBuffer.allocateDirect(message.length());
        in  = ByteBuffer.allocateDirect(message.length());
        worker = new Worker(new HelloWorldCallback(), 128);
    }

    protected void usage() {
        StringBuffer str = new StringBuffer();
        String sep = System.lineSeparator();
        str.append(sep + "Options:" + sep);
        str.append("\t-p <port>          TCP port (default: 12345)" + sep);
        str.append("\t-q                 print only errors (default: false)" + sep);
        str.append("\t-h                 display this help and exit" + sep);
        System.out.println(str.toString());
    }

    protected void parseArgs(String[] args) {
        options = new Options("p:hq", args);
        char opt;
        while ((opt = options.getOpt()) != Options.EOF) {
            String val = options.optArg;
            switch (opt) {
            case 'h':
                usage();
                System.exit(0);
                break;

            case 'p':
                port = Integer.parseInt(val);
                break;

            case 'q':
                quiet = true;
                break;

            default:
                System.out.println("Invalid option. Exiting...");
                usage();
                System.exit(1);
                break;
            }
        }
    }

    /**
     * Worker address receive protocol:
     * <ul>
     *  <li>Read remote address length from socket
     *  <li>Allocate byte[] to hold address
     *  <li>Read remote address from socket
     * </ul>
     */
    protected byte[] recvRemoteWorkerAddress(Socket sock) throws IOException {
        DataInputStream inStream = new DataInputStream(sock.getInputStream());
        int length = inStream.readInt();
        byte[] remoteWorkerAddress = new byte[length];
        inStream.readFully(remoteWorkerAddress);

        return remoteWorkerAddress;
    }

    /**
     * Worker address send protocol:
     * <ul>
     *  <li>Get local address from Worker
     *  <li>Write local address length to socket
     *  <li>Write local address to socket
     * </ul>
     */
    protected void sendLocalWorkerAddress(Socket sock) throws IOException {
        DataOutputStream outStream =
                new DataOutputStream(sock.getOutputStream());

        byte[] localWorkerAddress = worker.getWorkerAddress();
        outStream.writeInt(localWorkerAddress.length);
        outStream.write(localWorkerAddress);
        outStream.flush();
    }

    protected void send() throws IOException {
        out.put(message.getBytes());
        out.flip();
        ep.streamSend(out, 0);
    }

    protected void recv() throws IOException {
        ep.streamRecv(in, 1);
    }

    public void run(String[] args) throws IOException {
        parseArgs(args);
        try {
            init();
            exchangeWorkerAddress();
            exchangeData();
            conditionalPrint("Received:");
            in.flip();
            conditionalPrint(InBufferToString());
        } catch (IOException e) {
            System.out.printf("[Error] %s:\n", this.getClass().getSimpleName());
            throw e;
        } finally {
            close();
        }
    }

    protected void close() {
        // Free allocated native resources
        worker.close();
    }

    protected void conditionalPrint(String message) {
        if (!quiet) {
            System.out.println(message);
        }
    }

    private String InBufferToString() {
        StringBuffer str = new StringBuffer();
        while (in.hasRemaining()) {
            str.append((char) in.get());
        }

        return str.toString();
    }

    protected abstract void exchangeWorkerAddress() throws IOException;

    protected abstract void exchangeData() throws IOException;


    private class HelloWorldCallback extends TestCallback {
        @Override
        public void sendCompletionHandler(long requestId,
                                          CompletionStatus status) {
            sendComp = true;
        }

        @Override
        public void recvCompletionHandler(long requestId,
                                          CompletionStatus status,
                                          int length) {
            in.position(in.position() + length);
            recvComp = true;
        }
    }
}
