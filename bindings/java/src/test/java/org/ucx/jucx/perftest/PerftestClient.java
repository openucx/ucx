package org.ucx.jucx.perftest;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.LongStream;

import org.ucx.jucx.perftest.PerftestDataStructures.PerfMeasurements;
import org.ucx.jucx.perftest.PerftestDataStructures.PerfParams;
import org.ucx.jucx.perftest.PerftestDataStructures.PerfTestType;
import org.ucx.jucx.util.Time;

public interface PerftestClient {

	default public void printResults(PerfContext ctx) {
		PerfParams params = ctx.params;
		PerfMeasurements meas = ctx.measure;
		long[] results = meas.timeSamples();
		int iters = meas.iters;
		int size = params.size;
		Function<Double, String> printable = (d) -> new DecimalFormat("#0.000").format(d);
		OutputStreamWriter out;
		PrintWriter wr = null;
		try {
            out = new OutputStreamWriter(params.filename == null ?
                                        System.out : new FileOutputStream(params.filename));
            wr = new PrintWriter(out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }

		int factor = 1;
		long total = meas.endTime - meas.startTime;
		if (params.testType == PerfTestType.JUCX_TEST_PINGPONG) {
			factor = 2;
			total = LongStream.of(results).sum();
		}

		double[] percentile = { 0.99999, 0.9999, 0.999, 0.99, 0.90, 0.50 };

		printToFile(results, null, 20);
		Arrays.sort(results);

		String format = "%-25s = %-10s";
		wr.println(String.format(format, "---> <MAX> observation", printable.apply(Time.nanosToUsecs(results[results.length - 1]) / factor)));
		for (double per : percentile) {
			int index = (int)(0.5 + per*iters) - 1;
			wr.println(String.format(format, "---> percentile " + per, printable.apply(Time.nanosToUsecs(results[index]) / factor)));
		}
		wr.println(String.format(format, "---> <MIN> observation", printable.apply(Time.nanosToUsecs(results[0]) / factor)));

		wr.println();

		double secs = Time.nanosToSecs(total);
		double totalMBytes = (double)size * iters / Math.pow(2, 20);
		wr.println("average latency (usec): " + printable.apply(Time.nanosToUsecs(total) / iters / factor));
		wr.println("message rate (msg/s): " + (int)(iters/secs));
		wr.println("bandwidth (MB/s) : " + printable.apply(totalMBytes/secs));
		wr.close();
	}

	default public void printToFile(long[] arr, String filename, double lowerBound) {
		if (filename == null) {
            return;
        }
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i < arr.length; i++) {
				double l = Time.nanosToUsecs(arr[i]);
				if (l > lowerBound) {
					out.write(String.format("arr[%d] = %s", i, Double.toString(l)));
					out.newLine();
				}
			}
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
