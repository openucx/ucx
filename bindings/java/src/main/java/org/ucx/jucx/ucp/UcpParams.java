/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import static org.ucx.jucx.ucs.UcsBits.UCS_BIT;

/**
 * Tuning parameters for UCP library.
 *
 * <p>The structure defines the parameters that are used for
 * UCP library tuning during UCP library ucp_init "initialization".</p>
 *
 * <p>UCP library implementation uses the @UcpFeatures "features"
 * parameter to optimize the library functionality that minimize memory
 * footprint. For example, if the application does not require send/receive
 * semantics UCP library may avoid allocation of expensive resources associated with
 * send/receive queues.</p>
 */
public class UcpParams {

    @Override
    public String toString() {
        return "UcpParams{"
            + "fieldMask=" + fieldMask
            + ", features=" + features
            + ", tagSenderMask=" + tagSenderMask
            + ", mtWorkersShared=" + mtWorkersShared
            + ", estimatedNumEps=" + estimatedNumEps
            + '}';
    }

    /**
     * UCP context parameters field mask.
     *
     * <p>The enumeration allows specifying which fields in {@link UcpParams} are
     * present. It is used for the enablement of backward compatibility support.
     */
    public enum UcpParamField {
        UCP_PARAM_FIELD_FEATURES(UCS_BIT(0L)),
        // UCP_PARAM_FIELD_REQUEST_SIZE(UCS_BIT(1L)),
        // UCP_PARAM_FIELD_REQUEST_INIT(UCS_BIT(2L)),
        // UCP_PARAM_FIELD_REQUEST_CLEANUP(UCS_BIT(3L)),
        UCP_PARAM_FIELD_TAG_SENDER_MASK(UCS_BIT(4L)),
        UCP_PARAM_FIELD_MT_WORKERS_SHARED(UCS_BIT(5L)),
        UCP_PARAM_FIELD_ESTIMATED_NUM_EPS(UCS_BIT(6L));

        private long value;

        UcpParamField(long value) {
            this.value = value;
        }

        public long getValue() {
            return value;
        }
    }

    /**
     * UCP configuration features
     *
     * <p>The enumeration list describes the features supported by UCP.
     * An application can request the features using "UCP parameters"
     * during "UCP initialization" process.
     */
    public enum UcpFeature {
        UCP_FEATURE_TAG(UCS_BIT(0L)),    // Request tag matching support
        UCP_FEATURE_RMA(UCS_BIT(1L)),    // Request remote memory access support
        UCP_FEATURE_AMO32(UCS_BIT(2L)),  // Request 32-bit atomic operations support
        UCP_FEATURE_AMO64(UCS_BIT(3L)),  // Request 64-bit atomic operations support
        UCP_FEATURE_WAKEUP(UCS_BIT(4L)), // Request interrupt notification support
        UCP_FEATURE_STREAM(UCS_BIT(5L)); // Request stream support

        private long value;

        UcpFeature(long value) {
            this.value = value;
        }

        public long getValue() {
            return value;
        }
    }


    /**
     * Mask of valid fields in this structure, using bits from {@link UcpParamField}
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    public long fieldMask;

    /**
     * UCP ucp_feature "features" that are used for library
     * initialization. It is recommended for applications only to request
     * the features that are required for an optimal functionality
     * This field must be specified.
     */
    public long features;

    /**
     * Mask which specifies particular bits of the tag which can uniquely
     * identify the sender (UCP endpoint) in tagged operations.
     * This field defaults to 0 if not specified.
     */
    public long tagSenderMask;

    /**
     * This flag indicates if this context is shared by multiple workers
     * from different threads. If so, this context needs thread safety
     * support; otherwise, the context does not need to provide thread
     * safety.
     * For example, if the context is used by single worker, and that
     * worker is shared by multiple threads, this context does not need
     * thread safety; if the context is used by worker 1 and worker 2,
     * and worker 1 is used by thread 1 and worker 2 is used by thread 2,
     * then this context needs thread safety.
     * Note that actual thread mode may be different from mode passed
     * to ucp_init. To get actual thread mode use ucp_context_query.
     */
    public boolean mtWorkersShared;

    /**
     * An optimization hint of how many endpoints would be created on this context.
     * For example, when used from MPI or SHMEM libraries, this number would specify
     * the number of ranks (or processing elements) in the job.
     * Does not affect semantics, but only transport selection criteria and the
     * resulting performance.
     * The value can be also set by UCX_NUM_EPS environment variable. In such case
     * it will override the number of endpoints set by {@code estimatedNumEps}
     */
    public long estimatedNumEps;
}

