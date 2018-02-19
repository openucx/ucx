/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
package org.ucx.jucx.util;

public final class Time {

	private static final double SECOND     = 1000000000.0;
	private static final double U_SECOND   = 1000.0;

	private Time() {}

	public static long secsToNanos(double secs) {
		return (long)(secs * SECOND);
	}

	public static double nanosToUsecs(long time) {
		return time/U_SECOND;
	}

	public static double nanosToSecs(long time) {
		return time/SECOND;
	}
}
