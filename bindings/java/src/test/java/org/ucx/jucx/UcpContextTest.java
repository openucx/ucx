/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.ucx.jucx.ucp.UcpContext;
import org.ucx.jucx.ucp.UcpParams;

import java.util.concurrent.atomic.AtomicBoolean;

public class UcpContextTest {

    public static UcpContext createContext(UcpParams contextParams) {
        UcpContext context = Bridge.createContext(contextParams);
        assertTrue(context.getNativeId() > 0);
        return context;
    }

    public static void closeContext(UcpContext context) {
        context.close();
        assertEquals(context.getNativeId(), 0L);
    }

    @Test
    public void testCreateSimpleUcpContext() {
        UcpParams contextParams = new UcpParams();
        contextParams.features = UcpParams.UcpFeature.UCP_FEATURE_TAG.getValue();
        contextParams.fieldMask = UcpParams.UcpParamField.UCP_PARAM_FIELD_FEATURES.getValue();
        UcpContext context = createContext(contextParams);
        closeContext(context);
    }

    @Test
    public void testCreateUcpContextRdma() {
        UcpParams contextParams = new UcpParams();
        contextParams.features = UcpParams.UcpFeature.UCP_FEATURE_TAG.getValue()
            | UcpParams.UcpFeature.UCP_FEATURE_RMA.getValue();
        contextParams.estimatedNumEps = 10;
        contextParams.mtWorkersShared = false;
        contextParams.fieldMask =
            UcpParams.UcpParamField.UCP_PARAM_FIELD_ESTIMATED_NUM_EPS.getValue() |
                UcpParams.UcpParamField.UCP_PARAM_FIELD_MT_WORKERS_SHARED.getValue() |
                UcpParams.UcpParamField.UCP_PARAM_FIELD_FEATURES.getValue();

        UcpContext context = createContext(contextParams);
        closeContext(context);
    }
}
