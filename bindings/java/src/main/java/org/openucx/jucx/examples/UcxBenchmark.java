/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.examples;

import org.openucx.jucx.ucp.*;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public abstract class UcxBenchmark {

    protected static Map<String, String> argsMap = new HashMap<>();

    // Stack of closable resources (context, worker, etc.) to be closed at the end.
    protected static Stack<Closeable> resources = new Stack<>();

    protected static UcpContext context;

    protected static UcpWorker worker;

    protected static int serverPort;

    protected static int numIterations;

    protected static long totalSize;

    protected static UcpMemMapParams allocationParams;

    private static String DESCRIPTION = "JUCX benchmark.\n" +
        "Run: \n" +
        "java -cp jucx.jar org.openucx.jucx.examples.UcxReadBWBenchmarkReceiver " +
        "[s=host] [p=port] [n=number of iterations]\n" +
        "java -cp jucx.jar org.openucx.jucx.examples.UcxReadBWBenchmarkSender " +
        "[s=receiver host] [p=receiver port] [t=total size to transfer]\n\n" +
        "Parameters:\n" +
        "h - print help\n" +
        "s - IP address to bind sender listener (default: 0.0.0.0)\n" +
        "p - port to bind sender listener (default: 54321)\n" +
        "t - total size in bytes to transfer from sender to receiver (default 10000)\n" +
        "o - on demand registration (default: false) \n" +
        "n - number of iterations (default 5)\n";

    static {
        argsMap.put("s", "0.0.0.0");
        argsMap.put("p", "54321");
        argsMap.put("t", "10000");
        argsMap.put("o", "false");
        argsMap.put("n", "5");
    }

    /**
     * Initializes common variables from command line arguments.
     */
    protected static boolean initializeArguments(String[] args) {
        for (String arg: args) {
            if (arg.contains("h")) {
                System.out.println(DESCRIPTION);
                return false;
            }
            String[] parts = arg.split("=");
            argsMap.put(parts[0], parts[1]);
        }
        try {
            serverPort = Integer.parseInt(argsMap.get("p"));
            numIterations = Integer.parseInt(argsMap.get("n"));
            totalSize = Long.parseLong(argsMap.get("t"));
            allocationParams = new UcpMemMapParams().allocate().setLength(totalSize);
            if (argsMap.get("o").compareToIgnoreCase("true") == 0) {
                allocationParams.nonBlocking();
            }
        } catch (NumberFormatException ex) {
            System.out.println(DESCRIPTION);
            return false;
        }
        return true;
    }

    protected static void createContextAndWorker() {
        context = new UcpContext(new UcpParams().requestWakeupFeature()
            .requestRmaFeature().requestTagFeature());
        resources.push(context);

        worker = context.newWorker(new UcpWorkerParams());
        resources.push(worker);
    }

    protected static double getBandwithGbits(long nanoTimeDelta, long size) {
        double sizeInGigabits = (double)size * 8.0 / 1e9;
        double secondsElapsed = nanoTimeDelta / 1e9;
        return sizeInGigabits / secondsElapsed;
    }

    protected static void closeResources() throws IOException {
        while (!resources.empty()) {
            resources.pop().close();
        }
    }
}
