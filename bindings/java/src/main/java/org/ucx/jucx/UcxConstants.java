/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx;

import org.ucx.jucx.ucp.UcpParams;

public class UcxConstants {
    static {
        NativeLibs.load();
        loadConstants();
    }

    /**
     * UCP context parameters field mask.
     *
     * <p>The enumeration allows specifying which fields in {@link UcpParams} are
     * present. It is used for the enablement of backward compatibility support.
     */
    public static long UCP_PARAM_FIELD_FEATURES;
    public static long UCP_PARAM_FIELD_TAG_SENDER_MASK;
    public static long UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    public static long UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;

    /**
     * UCP configuration features
     *
     * <p>The enumeration list describes the features supported by UCP.
     * An application can request the features using "UCP parameters"
     * during "UCP initialization" process.
     */
    public static long UCP_FEATURE_TAG;
    public static long UCP_FEATURE_RMA;
    public static long UCP_FEATURE_AMO32;
    public static long UCP_FEATURE_AMO64;
    public static long UCP_FEATURE_WAKEUP;
    public static long UCP_FEATURE_STREAM;

    private static native void loadConstants();
}
