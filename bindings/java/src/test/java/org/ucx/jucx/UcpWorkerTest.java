/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx;

import org.junit.Test;
import org.ucx.jucx.ucp.UcpContext;
import org.ucx.jucx.ucp.UcpParams;
import org.ucx.jucx.ucp.UcpWorker;
import org.ucx.jucx.ucp.UcpWorkerParams;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.BitSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class UcpWorkerTest {
  static int numWorkers = Runtime.getRuntime().availableProcessors();

  public static UcpWorker createWorker(UcpWorkerParams workerParams, UcpContext context) {
    UcpWorker worker = Bridge.createWorker(workerParams, context);
    assertTrue(worker.getNativeId() > 0);
    return worker;
  }

  @Test
  public void testMultipleWorkersInitialization() {
    UcpParams contextParams = new UcpParams();
    contextParams.features = UcpParams.UcpFeature.UCP_FEATURE_TAG.getValue();
    contextParams.fieldMask = UcpParams.UcpParamField.UCP_PARAM_FIELD_FEATURES.getValue();
    UcpContext context = UcpContextTest.createContext(contextParams);

    UcpWorkerParams workerParams = new UcpWorkerParams();
    workerParams.fieldMask =
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_CPU_MASK.getValue() |
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_THREAD_MODE.getValue();
    workerParams.threadMode = UcxTools.UcsTreadMode.UCS_THREAD_MODE_SINGLE;

    for (int i = 0; i < numWorkers; i++) {
      BitSet cpuMask = new BitSet();
      cpuMask.set(i);
      workerParams.cpuMask = cpuMask;
      UcpWorker worker = createWorker(workerParams, context);
      worker.release();
      assertEquals(worker.getNativeId(), 0L);
    }
    UcpContextTest.closeContext(context);
  }

  @Test
  public void testMultipleWorkersFromMultipleContexts() {
    UcpWorker workers[] = new UcpWorker[numWorkers];

    UcpParams tcpContextParams = new UcpParams();
    tcpContextParams.features = UcpParams.UcpFeature.UCP_FEATURE_TAG.getValue();
    tcpContextParams.fieldMask = UcpParams.UcpParamField.UCP_PARAM_FIELD_FEATURES.getValue();
    UcpContext tcpContext = UcpContextTest.createContext(tcpContextParams);
    UcpWorkerParams tcpWorkerParams = new UcpWorkerParams();
    tcpWorkerParams.fieldMask =
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_CPU_MASK.getValue() |
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_THREAD_MODE.getValue() |
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_USER_DATA.getValue();
    tcpWorkerParams.userData = ByteBuffer.allocateDirect(100).order(ByteOrder.nativeOrder());
    tcpWorkerParams.threadMode = UcxTools.UcsTreadMode.UCS_THREAD_MODE_MULTI;

    UcpParams rdmaContextParams = new UcpParams();
    rdmaContextParams.features = UcpParams.UcpFeature.UCP_FEATURE_TAG.getValue() |
      UcpParams.UcpFeature.UCP_FEATURE_RMA.getValue();
    rdmaContextParams.fieldMask = UcpParams.UcpParamField.UCP_PARAM_FIELD_FEATURES.getValue();
    UcpContext rdmaContext = UcpContextTest.createContext(rdmaContextParams);
    UcpWorkerParams rdmaWorkerParams = new UcpWorkerParams();

    rdmaWorkerParams.fieldMask =
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_CPU_MASK.getValue() |
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_THREAD_MODE.getValue() |
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_EVENTS.getValue()  |
      UcpWorkerParams.UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_USER_DATA.getValue();
    rdmaWorkerParams.userData = ByteBuffer.allocateDirect(100).order(ByteOrder.nativeOrder());
    rdmaWorkerParams.threadMode = UcxTools.UcsTreadMode.UCS_THREAD_MODE_SINGLE;
    rdmaWorkerParams.events = UcpWorkerParams.UcpWakeupEventTypes.UCP_WAKEUP_RMA.getValue();

    for (int i = 0; i < numWorkers; i++) {
      BitSet cpuMask = new BitSet();
      cpuMask.set(i);
      if (i % 2 == 0) {
        tcpWorkerParams.cpuMask = cpuMask;
        tcpWorkerParams.userData.asCharBuffer().put("TCP WORKER" + i);
        workers[i] = createWorker(tcpWorkerParams, tcpContext);
      } else {
        rdmaWorkerParams.cpuMask = cpuMask;
        rdmaWorkerParams.userData.asCharBuffer().put("RDMA WORKER" + i);
        workers[i] = createWorker(rdmaWorkerParams, rdmaContext);
      }
    }

    for (int i = 0; i < numWorkers; i++){
      workers[i].release();
      assertEquals(workers[i].getNativeId(), 0L);
    }
    tcpContext.close();
    rdmaContext.close();
  }

}
