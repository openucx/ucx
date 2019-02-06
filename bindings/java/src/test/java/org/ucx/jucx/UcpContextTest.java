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

  public static UcpContext createContext(UcpParams contextParams){
    UcpContext context = Bridge.createContext(contextParams);
    assertTrue(context.getNativeId() > 0);
    return context;
  }

  public static void closeContext(UcpContext context){
    context.close();
    assertEquals(context.getNativeId(), 0L);
  }

  @Test
  public void testCreateSimpleUcpContext(){
    UcpParams contextParams = new UcpParams();
    contextParams.features = UcpParams.UcpFeature.UCP_FEATURE_TAG.getValue();
    contextParams.fieldMask = UcpParams.UcpParamField.UCP_PARAM_FIELD_FEATURES.getValue();
    UcpContext context = createContext(contextParams);
    closeContext(context);
  }

  @Test
  public void testCreateUcpContextRdma(){
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


  @Test(expected = AssertionError.class) // TODO: make callbacks in C to propagate to java
  public void testRequestsCallbacks(){
    UcpParams contextParams = new UcpParams();
    contextParams.features = UcpParams.UcpFeature.UCP_FEATURE_TAG.getValue();
    contextParams.fieldMask =
      UcpParams.UcpParamField.UCP_PARAM_FIELD_FEATURES.getValue() |
        UcpParams.UcpParamField.UCP_PARAM_FIELD_REQUEST_SIZE.getValue() |
        UcpParams.UcpParamField.UCP_PARAM_FIELD_REQUEST_INIT.getValue() |
        UcpParams.UcpParamField.UCP_PARAM_FIELD_REQUEST_CLEANUP.getValue();
    contextParams.requestSize = 100;
    String message = "Test request data";
    AtomicBoolean initCallbackCalled = new AtomicBoolean(false);
    contextParams.requestInit = data -> {
      data.asCharBuffer().put(message);
      assertEquals(data.capacity(), 100);
      initCallbackCalled.set(true);
    };

    AtomicBoolean cleanupCallbackCalled = new AtomicBoolean(false);
    contextParams.requestCleanup = data -> {
      assertEquals(data.asCharBuffer().toString(), message);
      cleanupCallbackCalled.set(true);
    };
    UcpContext context = createContext(contextParams);

    assertTrue(initCallbackCalled.get());
    closeContext(context);

    assertTrue(cleanupCallbackCalled.get());
  }

}
