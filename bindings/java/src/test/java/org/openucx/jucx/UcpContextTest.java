/*
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.openucx.jucx.ucp.UcpContext;
import org.openucx.jucx.ucp.UcpParams;

public class UcpContextTest {

    public static UcpContext createContext(UcpParams contextParams) {
        UcpContext context = new UcpContext(contextParams);
        assertTrue(context.getNativeId() > 0);
        return context;
    }

    public static void closeContext(UcpContext context) {
        context.close();
        assertEquals(context.getNativeId(), null);
    }

    @Test
    public void testCreateSimpleUcpContext() {
        UcpParams contextParams = new UcpParams().requestTagFeature();
        UcpContext context = createContext(contextParams);
        closeContext(context);
    }

    @Test
    public void testCreateUcpContextRdma() {
        UcpParams contextParams = new UcpParams().requestTagFeature().requestRmaFeature()
            .setEstimatedNumEps(10).setMtWorkersShared(false).setTagSenderMask(0L);
        UcpContext context = createContext(contextParams);
        closeContext(context);
    }

    @Test
    public void testConfigMap() {
        UcpParams contextParams = new UcpParams().requestTagFeature()
            .setConfig("TLS", "abcd").setConfig("NOT_EXISTING_", "234");
        boolean catched = false;
        try {
            createContext(contextParams);
        } catch (UcxException exception) {
            assertEquals("No such device", exception.getMessage());
            catched = true;
        }
        assertTrue(catched);

        // Return back original config
        contextParams = new UcpParams().requestTagFeature().setConfig("TLS", "all");
        UcpContext context = createContext(contextParams);
        closeContext(context);
    }
    
    @Test(expected = NullPointerException.class)
    public void testCatchJVMSignal() {
        UcpParams contextParams = new UcpParams().requestTagFeature();
        UcpContext context = createContext(contextParams);
        closeContext(context);
        long nullPointer = context.getNativeId();
        nullPointer += 2;
    }
}
